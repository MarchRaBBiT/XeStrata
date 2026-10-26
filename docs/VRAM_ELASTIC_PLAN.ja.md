<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# HIP の vram_elastic 対応計画

[English](VRAM_ELASTIC_PLAN.md) | 日本語

状態: 手順 1 で止めました（2026-10-26）。
AMD の GPU（ROCm）の仮想メモリーは、unmap したアドレスへ別の物理メモリーを map し直すと、書いたデータが別のメモリーへ行くことがあるので、HIP の `vram_elastic` は進めません。
確かめた結果は[手順 1 の結果](#手順-1-の結果)に、もう一度始めるときの条件は[再開の条件](#再開の条件)にあります。手順 2〜6 は行っていません。

この計画の目的は、既存の `vram_elastic` を SYCL の HIP 経路と resident モードで使えるようにし、
文章のエンジンを再起動せず、画像に必要な間だけエキスパートのキャッシュから VRAM を返して取り戻すことでした。
現在の使い方は [DETAILS](DETAILS.ja.md#ほかのプログラムのために-vram-を空ける) にあります。

対象マシン、GPU、ツールチェーン、コンテナ、設定、サービスの操作方法、機能照会の結果は、
リポジトリ直下の `LOCAL.md` を参照します。この計画にはマシン固有の値を持たせません。
測定は既存の記録を先に読み、新しいバイナリーや設定で必要な差だけを測り、`<data>/records/` に残します（`<data>`: `LOCAL.md`）。

## 目的と範囲

- `POST /v1/vram` で、要求の合間にキャッシュを縮小・復元できるようにします。
- RAM の写しを持つ resident モード（`--resident-experts`、RAM が少ないときの自動移行、`--resident-budget-gib`、`--resident-cpu-experts`）でも、既存の RAM とファイルのエキスパートを使います。
- 画像入力では、`vision.elastic` でエンコーダーを必要な間だけ起動します。
- 画像生成の `gpu_policy: shrink` は、画像入力とは別に容量と速度を確認してから適用します。

API や設定キーの追加、複数 GPU とリモートキャッシュへの対応、常時 VRAM を監視して勝手に縮小する仕組み、
エキスパート全体の RAM への強制コピーは、この改修の範囲に含めません。

## 手順 1 の結果

Radeon VII（gfx906）で、HIP の `hipMem*` と、その下の ROCr の `hsa_amd_vmem_*` を直接呼ぶプローブを作り、
キャッシュが行う操作を SYCL のキューから出して確かめました。
ROCm は 7.14（gfx906 を含めてビルドしたもの）、カーネルは Linux 7.0、SYCL 側は intel/llvm v7.2.0 です。
v7.2.0 の UR は HIP の VMM を持たないので、XeStrata の経路（`sycl_ext_oneapi_virtual_mem`）は通していません。

正しく動いたもの:

- 能力照会は対応と答え、粒度は最小 4 KiB、推奨 2 MiB です。0 以外の offset での map は `hipErrorInvalidValue` になります（XeStrata は offset 0 で map します）。
- 一度も map していないアドレスでは、pinned と pageable の host との両方向のコピー、device メモリーとのコピー、memset、fill、カーネルの読み書き、graph の再生がすべて一致しました。
  2 つの物理割当をまたぐコピー、memset、カーネルも一致しました。
- map したアドレスに対する `sycl::get_pointer_type` は device を、予約しただけのアドレスは unknown を返します（UR の assert には当たりません）。
- map したまま `hipMemRelease` すると、メモリーは unmap まで使え、unmap で解放されます。
- 512 MiB のセグメント 4 つで縮小・復元を 20 回繰り返した後、HIP とカーネル側の両方でリークはありませんでした（縮小と復元の間を 100 ms 空けた場合）。
- 速度は通常の device メモリーと同じでした（host とのコピー 7.1 GB/s、2 GiB のカーネル読み出し 871 GB/s）。
  512 MiB のセグメントの割当・map・アクセス権の設定は 0.1 ms、unmap・解放は 0.2 ms です。
- VRAM を使い切ると、`hipMemCreate` は `hipErrorOutOfMemory` を返し、それまでの割当の内容は保たれます。

止めた理由:

- unmap したアドレスへ別の物理メモリーを map してすぐ書くと、書き込みが前に map していた物理メモリーへ行きます（2 MiB で 10 回中 6〜10 回）。
  unmap と map の間を 1 ms 以上空けると起きませんでした。
- そのアドレスに書いた後、隣のアドレスへ別の物理メモリーを map すると、前のアドレスが隣の新しいメモリーを指すことがあります。
  前のアドレスに書いた内容は失われ、隣と同じメモリーを読み書きします（2 MiB で 10 回中 0〜4 回、512 MiB では unmap と map の間を 100 ms 空けても 5 回中 1 回）。
- ROCr の `hsa_amd_vmem_*` を直接呼んでも同じなので、HIP のランタイムより下（ROCr の仮想メモリーか、カーネルのドライバー）の問題です。
  UR のパッチでも XeStrata の側でも直せません。
- `vram_elastic` の復元は、縮小で unmap したアドレスへ新しい物理メモリーを map し、続くセグメントをその隣に map します。
  `--kv-grow` は、キャッシュから外した物理メモリーをすぐ K/V のアドレスへ map し、返すときは別の物理メモリーをキャッシュのアドレスへ map し直します。
  どちらもこの手順に当たり、エキスパートや K/V の中身がエラーを出さずに壊れます。

空き容量について分かったこと（手順 4 の前提）:

- 解放した VRAM は、HIP の空きにはすぐ戻り、カーネル側の数え方では 40 ms 後に戻ります（`hipMemRelease` と `hipFree` のどちらも）。
- 同じ GPU で Vulkan のエンコーダーが約 1 GiB を使っていても、HIP の空きは減りません。
  HIP の空きに合わせて割り当てると、エンコーダーの約 870 MiB が GTT へ追い出され、1 枚のエンコードは 1.37 秒から 5.5 秒になりました。
  VRAM を空けても、エンコーダーを起動し直すまで速さは戻りません。

## 再開の条件

- 別の物理メモリーへの map し直しと、その隣への map を、unmap の直後と間を空けた場合の両方で正しく扱う ROCm（ROCr とカーネルのドライバー）が出たら、手順 1 をやり直し、通れば手順 2 から始めます。
- intel/llvm の sycl ブランチには HIP の VMM の実装が入りました（c13ad6c7、#22671）。
  これを含む intel/llvm に上げると、HIP のデバイスが仮想メモリーに対応すると答え、粒度が 2 MiB なので、指定しない起動でも `--kv-grow` が有効になります（[DETAILS](DETAILS.ja.md#kv-キャッシュの置き場所と精度)）。
  上げる前に、この不具合のある実行時ライブラリーでは仮想メモリーを使わない判定を入れます。
- HIP に依らない部分（resident モードでの `vram_elastic`、カーネル側の空きによる容量の判定）は、ほかの GPU でも意味がありますが、まだ始めていません。

## 実装上の根拠

| 箇所 | 現在の制約と対応 |
| --- | --- |
| intel/llvm v7.2.0 の UR HIP アダプター | `device.cpp` の VMM 能力照会は false 固定で、`virtual_mem.cpp` と `physical_mem.cpp` は未実装です。intel/llvm の sycl ブランチには実装があり（c13ad6c7、#22671）、v7.2.0 の木にそのまま当たります。補うときは、既存の番号付きパッチ機構でこれを移します。 |
| [generate.cpp](../src/program/generate.cpp) の `vram_elastic` 判定 | resident モード（`resident_cpu_experts` か `resident_gib`）を禁止しています。ファイルへの読み出しと復元の安全性を試験してから制約を緩めます。 |
| `generate.cpp` の `--kv-grow` 判定 | `resident_cpu_experts` だけを見ており、RAM が少ないときの resident への自動移行より前に決まります。HIP で VMM が使えるようになると、指定しない起動でも resident モードで `--kv-grow` が入り得ます。 |
| [ExpertCache](../include/strata/core/expert_cache.hpp) | graph、verify plan、prefill がキャッシュのアドレスを保持します。予約した仮想アドレスを維持し、物理メモリーだけを付け外しします。 |
| [FileExpertSource](../src/core/expert_source.cpp) | RAM にないエキスパートは既存のファイル読み出しで補えます。RAM の写しは GPU にないエキスパートと prefill の貸出枠のエキスパートを持つので、縮小で貸出枠が写しのない枠へ移ると、prefill がチャンクごとにファイルを読みます（upstream #669）。縮小に合わせて RAM の写しを増やす設計にはしません。 |
| `generate.cpp` の `vram_command` にある grow | `blob_to_copy` の一時バッファーを queued 転送の完了前に再利用し得ます。既存 prefill の refill と同じ寿命管理にします。 |
| `generate.cpp` の `vram_command` の容量計算 | 縮小量と答えの `vram_free_mib` を `gpu::mem_info`（HIP では hipMemGetInfo）から求めます。HIP の空きは、同じ GPU のほかのプログラムの割当を引きません（手順 1 で Vulkan の割当について確認）。 |
| [gpu.cpp](../src/core/gpu.cpp) と [expert_cache.cpp](../src/core/expert_cache.cpp) | `vmem_unmap` の失敗を shrink が受け取れません。解放に失敗した領域を解放済みと報告しないようにします。 |
| [server.py](../serve/server.py) の `_vision_up` と `vision_attempts` | elastic のエンコーダーは起動時に作らないので、設定の fallback は起動時にしか試されません。画像の番で起動に失敗したエンコーダーは、server を再起動するまで止まったままです。 |

## 実装手順

手順 2 以降は、手順 1 が通らなかったので行っていません。[再開の条件](#再開の条件)を満たしたときの手順として残します。

### 1. HIP VMM の実動作を確かめる

[AMD の VMM 手順](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html)
に沿って、まず小さい領域で能力照会、最小・推奨粒度、仮想アドレスの予約、物理割当、map、アクセス権、
読み書き、unmap、解放、同じ仮想アドレスへの再割当を確認します。
読み書きは、XeStrata がキャッシュに行う操作で確かめます。pinned と pageable の host からのコピー、device から host への写し戻し、
memset、カーネルからの読み書き、graph の再生です。
VMM のアドレスに対するポインターの種類の照会（`sycl::get_pointer_type`）と、map したままの物理メモリーを解放したときの振る舞いも確かめます。
対応フラグだけでは次へ進まず、データの一致とリークのない解放までを合格条件にします。
失敗した場合は原因を切り分け、通常のキャッシュのままにします。

HIP の空き VRAM と同じ GPU のカーネル側の表示も比較し、割当・解放への追従、
外部の Vulkan エンコーダーとの同居、GTT への退避を確認します。
対象の PCI アドレスや試験環境は `LOCAL.md` から取ります。

再割当は、unmap の直後と間を空けた場合の両方で、同じ物理メモリーと別の物理メモリーについて確かめ、その後に隣のアドレスへ map したときの前のアドレスの内容も確かめます。
2026-10-26 の結果は[手順 1 の結果](#手順-1-の結果)にあります。

### 2. UR HIP アダプターに VMM を実装する

intel/llvm の sycl ブランチにある HIP の VMM の実装（c13ad6c7、#22671）を移します。
`third_party/main/intel-llvm/patches/` に `NN-hip-virtual-memory.patch` を追加し、
UR の `device.cpp`、`physical_mem.hpp/.cpp`、`virtual_mem.cpp` を対象にします。
パッチの先頭には、既存のパッチと同じく SPDX の行と、何を直すかの説明を書きます。
HIP の能力照会、粒度、デバイスの選択、アクセス権、ハンドルの参照数、エラー変換、途中失敗時の解放が、
CUDA アダプターと同じ扱いになっていることを確かめます。非対応のデバイスは非対応と答えます。

XeStrata の core は既存の `sycl_ext_oneapi_virtual_mem` を使い、HIP API は UR 内に置きます。
セグメントの大きさはデバイスが報告する粒度と既存の設定から決め、GPU 名による分岐を加えません。
[intel/llvm のビルド手順](DEVTOOLS.ja.md#intelllvm-をソースからビルドする)に従い、
free と contrib-llvm の両ツールチェーンへパッチを適用します。
`tools/intel_llvm_build.py` は UR の試験をビルドしないので、UR の VMM の conformance 試験
（`unified-runtime/test/conformance/virtual_memory`）は、試験付きの UR を別にビルドして実行します。
ビルドできない場合は手順 1 の確認と `vmm_test` で代え、そのことを報告に書きます。
修正の説明は、`docs/DEVTOOLS.ja.md` と `docs/DEVTOOLS.md` の intel/llvm の節に、日本語から先に足します。

### 3. resident モードと縮小・復元を安全に接続する

`generate.cpp` の resident 制約は、次の試験が通る実装と一緒に緩めます。
単一 GPU、serve、リモートキャッシュなしなど、今回扱わない制約は維持します。

- 縮小で外れたエキスパートは `FileExpertSource` の RAM、なければファイルから読みます。
- RAM の写しは、GPU にないエキスパートと prefill の貸出枠のエキスパートだけを持つ状態を保ちます。
  grow の補充で、adaptive tier が戻した枠に代わりのエキスパートを選ぶときは、RAM の写しにあるもの（`has_resident`）を選びません。
- 縮小で貸出枠が RAM の写しのない枠へ移ると、prefill はチャンクごとにそのエキスパートをファイルから読みます。
  縮小後の短い・長い prefill の速度を測り、それを許容するか、貸出を小さくするかを決めます。
  RAM の写しを増やすのは、測定でほかに手がないと分かったときだけにします。
- grow では `srcp->transient()` を確認し、一時バッファーを使う転送は `fill_slot_blocking`、それ以外は queued とします。
- adaptive swap の完了待ちと GPU の同期の後に領域を変更し、`host_res` とデバイスの residency を更新します。
- prefill の貸出領域は残った枠へ移し、必要なら既存の処理でチャンクを小さくします。
- `--kv-grow` の判定を RAM が少ないときの resident への自動移行の後で行い、resident の条件を
  `vram_elastic` の判定（`resident_cpu_experts` か `resident_gib`）と揃えます。
  resident モードでも `--kv-grow` を許すなら、その理由と速度を試験で示します。
- map、unmap、転送、同期の途中失敗でも、実際に使える領域と読み終えた枠だけを公開します。
  復旧できないデバイスのエラーは要求へ返し、不完全な状態で推論を続けません。

追加の RAM 常駐表やキャッシュの配置表は、試験で必要と分かるまで作りません。
変更候補は `generate.cpp`、`gpu.cpp` とそのヘッダー、`expert_cache.cpp` とそのヘッダー、`expert_source.cpp` とそのヘッダーです。

### 4. 空き容量とサーバーの判定を合わせる

HIP とカーネル側の空き表示の差を切り分け、外部の割当を見落とさない容量判定にします。
`vram_command` は縮小量と答えの `vram_free_mib` を `gpu::mem_info` から求めるので、エンジン側の計算から直します。
Linux 側の補助的な照会が必要なら `src/platform/` を使い、対象 GPU を PCI アドレスで対応付けます。
表示が取れない場合の扱いも定め、ゼロや全容量を空きの実測値として代入しません。
VMM の割当を含め、VRAM と最大割当の制限を守ることも確認します。

[server.py](../serve/server.py) の `_vision_up` と [images.py](../serve/images.py) の `engine_elastic` は、
起動引数だけでなくエンジンが返す有効状態を使います。`INFO vram_elastic=0` を起動引数で覆しません。
`POST /v1/vram` の 200 応答だけでは必要量を確保できたと判断しません。
エンジンの答えの空きはエンジン自身の照会の値なので、server は既存の
[telemetry.py](../serve/telemetry.py) が読むカーネル側の空き（AMD では sysfs）でも確かめます。
エンコーダー起動前には、固定時間の sleep だけに頼らず、期限とキャンセルを持つ空き待ちを行います。

画像の番で必要量が空かないとき、または GPU のエンコーダーが起動できないときに、
設定の fallback のエンコーダーへ切り替える処理を新しく作ります。
切り替えたことは既存の `vision_fallback` で示します。fallback がなければ理由を返し、過剰な割当を始めません。
失敗・キャンセル後も予約とキャッシュを復元します。

### 5. テストとビルドを通す

変更を表す小さい失敗テストを先に追加し、実装後に次の条件を確認します。
使える既存の試験と fixtures は使い、新しい非自明な分岐には正常系と失敗系の試験を残します。
カバレッジは、ホストの C++ と Python の追加・変更部分で 80% 以上にします。
リポジトリにはカバレッジの計測がまだないので、道具と手順を実装の初めに決め、道具を足すなら DEVTOOLS に書きます。
SYCL のデバイスコードと GPU でしか通らない経路は計測の対象外とし、下の試験の合否で確かめます。

| 層 | 試験と合格条件 |
| --- | --- |
| VMM | UR の conformance 試験（手順 2）と [vmm_test](../tests/core/vmm_test.cpp)。対象 HIP GPU で skip せず、読み書きと再 map が一致し、メモリーが戻ること。 |
| キャッシュ | segmented cache の試験を `tests/core/` と `CMakeLists.txt` に追加。可変 blob サイズ、セグメント境界、最小サイズ、部分的な grow、失敗時の整合性を確認すること。 |
| resident | [file_expert_source_test](../tests/core/file_expert_source_test.cpp) を拡張。RAM の補完領域と、そこにない GGUF エキスパートのファイル読み出しを確認すること。 |
| エンジンの統合 | エンジンの subprocess に VRAM コマンドを送り、grow の一時バッファーの寿命、adaptive swap 後の復元、grow の補充が RAM の写しにあるエキスパートを選ばないこと、縮小後の短い・長い prefill（resident モードを含む）を確認すること。CTest にはエンジンを動かすモデルの fixture がないので、使うモデルを実装の初めに決め、CTest に入れられない試験は手順と結果を報告に残すこと。source の単体試験だけで済ませないこと。 |
| サーバー | `serve/` の unittest に能力判定、容量不足、解放の遅延、同時要求、キャンセル、画像の番での fallback への切り替え、復元失敗を追加。200 応答と目標量の達成を区別すること。 |
| E2E | 文章だけ、最初の画像入力、連続画像入力、画像後の会話、idle 後の復元を通すこと。通常の縮小・復元で文章のエンジンを再起動しないこと。 |
| 回帰 | free と contrib-llvm のビルド・CTest、NVIDIA の実行、小容量・小さい最大割当・行列エンジンなし・少ない CPU スレッドでの実行、Intel GPU の既存経路、パッチのない intel/llvm の実行時ライブラリー（`vram_elastic` が無効のまま動くこと）、Windows クロスビルドを確認すること。 |

VMM の能力照会を有効にすると、デバイスの粒度が 2 MiB 以上のとき `--kv-grow` が自動で有効になります。
`vram_elastic` を指定しない起動、`--kv-grow` の自動・明示・無効の各設定を、resident モード（自動移行を含む）とそれ以外の両方で回帰試験に含めます。
HIP でキャッシュを VMM のチャンクに置いたときの prefill と decode を置かないときと比べ、
遅くなるなら自動の既定をデバイスの報告による条件で決めます。GPU の名前や作り手では分けません。
メモリーと未定義動作に関係する変更には、別ビルドで address/undefined sanitizer と CTest を実行します。
文書は DETAILS の VRAM の節（HIP と resident モードの記述）を、コメントは `gpu.hpp` と `vmm.hpp` の対応アダプターの記述を、実装に合わせて直します。
提出前に `tools/lint/run.sh`、ライセンスの分類、`reuse lint` を通し、新しい警告や指摘を残しません。
試験結果と実行できなかった条件は実装の報告に残します。

### 6. 画像入力から段階的に導入する

最初に手動の縮小・復元で `INFO vram_elastic=1`、キャッシュの大きさ、空き VRAM、推論の復帰を確認します。
その後、設定の `vram_elastic: true` と `vision.elastic: true` を有効にします。
エンコーダー起動時の最大使用量とセグメントへの丸めを確認して `vision.elastic_mib` を決め、
従来の常時予約は、それをエンコーダーなしで運用できると確認してから減らします。
RAM の使用量、swap、GTT、最初の画像の待ち時間、prefill と decode、キャッシュを戻す時間を比較します。

画像生成は、その画像に必要な量と文章の固定領域が同時に収まることを別に確認してから `gpu_policy: shrink` を試します。
初期導入では既存の `unload` を維持し、収まらない設定ではそれを使います。
VMM 対応だけで画像生成全体が同居できるとは判断しません。

## 完了条件と戻し方

- HIP VMM が対象デバイスの能力に従って選ばれ、resident モードでも `vram_elastic` が有効になります。
- 縮小・復元で保持する枠のアドレスが変わらず、復元後のエキスパートの内容と推論が正しく戻ります。
- 目標まで空けられない場合、途中失敗、キャンセルでも、実際の容量と residency が一致します。
- 画像のない間はエンコーダーの VRAM をキャッシュへ返し、通常の画像入力で文章のエンジンを再起動しません。
- 指定しない起動での `--kv-grow` の自動の振る舞いが、手順 3 と 5 で決めたとおりです。
- 機能を指定しない起動と既存の対応 GPU で回帰がなく、必要なビルド・試験が合格します。

導入前に設定とバイナリー・ランタイムの組を保存します（置き場所: `LOCAL.md`）。
戻すときは `vram_elastic` と `vision.elastic` を無効にし、元の常時予約と画像生成の `unload` を戻して再起動します。
ランタイムにも問題がある場合は、保存したバイナリーとランタイムの組に戻します。
縮小中は RAM にないエキスパートのファイル読み出しが増え、prefill と decode が遅くなる可能性があるため、
正しさに加えて運用時の速度を採用条件にします。
完了したら使い方を DETAILS に書き、この計画と DETAILS からのリンクを外します。
