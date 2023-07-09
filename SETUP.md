# mdx での Megatron-DeepSpeed のセットアップ方法

## pyenv + venv でのセットアップ

基本的な手順は[この記事](https://zenn.dev/turing_motors/articles/04c1328bf6095a)と同様です。

1. pyenv が環境にあるかの確認
    ```bash
    > pyenv --version
    pyenv 2.3.21
    ```

    入っていない場合は `curl https://pyenv.run | bash`でinstall可能です。

    mdxでは、`~/.bashrc`が自動で読み込まれないようなので、pyenvをinstallした際は
    ```bash
    # pyenv
    export PYENV_ROOT="$HOME/.pyenv"
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    eval "$(pyenv virtualenv-init -)"
    ```
    を `~/.bashrc` に追記すると思いますが、手動で `source ~/.bashrc` する必要があります。

    (ログインするたびに、`bashrc`を読み込むのは手間ですが、お願いします)

2. pyenv で python を install
    ```bash
    > pyenv install 3.10.10
    > cd Megatron-DeepSpeed
    > pyenv local 3.10.10
    > python -m venv .env
    > source .env/bin/activate
    ```
    で、pythonの環境を作成します。

    この際、`pyenv local` で指定したpythonのバージョンが、`python --version` で表示されることを確認してください。

3. pip install

    `nvcc --version`で表示されるcudaのバージョンに合わせて、`requirements.txt`を変更してください。

    Megatron-DeepSpeedを`git@github.com:llm-jp/Megatron-DeepSpeed.git`からcloneしてきた場合は、
    ```bash
    git switch hpc/fujii/deepspeed-multi-node
    pip install -r requirements.txt
    ```

    とすることで、CUDA11.8に対応したPyTorchなどがinstallされます。

4. apex install

    NVIDIA:apex を install します。
    ```bash
    git clone git@github.com:NVIDIA/apex.git
    cd apex
    ```

    ここで apex を install 際のコマンドを [こちら](https://github.com/NVIDIA/apex#linux)から確認ください。pip versionによってコマンドが異なります。

Setup 完了です。

## conda でのセットアップ（非推奨）

注意：過去のテスト用に構築された環境のため、新しく環境構築する場合は、上記の pyenv + venv を使ってください。

```bash
source /model/share/miniforge/etc/profile.d/conda.sh
conda activate megatron-deepspeed
```

## Multi-Node 学習のための準備

### ssh config

`~/.ssh/config` に使用する node に `ssh <name>` で接続できるように `config` を設定してください。

ユーザー名や秘密鍵名、node の IP アドレスなどは変更する必要がありますが、以下が参考になると思います。

```bash
Host mdx
  HostName llm-jp.zapto.org
  User kazukifujii
  ServerAliveInterval 15
  IdentityFile ~/.ssh/mdx

Host 10.2.72.135
  HostName 10.2.72.135
  User kazukifujii
  IdentityFile ~/.ssh/mdx
  ServerAliveInterval 15
  ProxyCommand ssh -W %h:%p mdx

Host 10.2.72.136
  HostName 10.2.72.136
  User kazukifujii
  IdentityFile ~/.ssh/mdx
  ServerAliveInterval 15
  ProxyCommand ssh -W %h:%p mdx
```

mdx に login した状態で `ssh <node-name>`で接続できることを確認してください。


### mpirun

[このファイル](https://github.com/llm-jp/Megatron-DeepSpeed/blob/hpc/fujii/deepspeed-multi-node/scripts/mpirun/345m_dp16.sh)を参考にしてください。

`-H `には、使用するノードの名前を記入してください。(自分は、`.ssh/config` の HostName と Host名が同じなので)
なお、ノードの名前をスクリプトに記述する代わりに、hostfile を使う方法もあります。詳細は下記の Appendix を参照してください。

```bash
mpirun -np $WORLD_SIZE --npernode $GPUS_PER_NODE \
  -H 10.2.72.135:8,10.2.72.136:8 \
  -x MASTER_ADDR=10.2.72.135 \
```

とします。MASTER_ADDR は、`-H` で指定したノードのうち、一つを指定してください。
基本的には、今ログインしているノードを指定すれば良いと思います。

上の場合では、`10.2.72.135`が今ログインしているnodeです。


`10.2.72.135`にて、以下のようにjobを投げます。

```bash
bash scripts/mpirun/345m_dp16.sh
```

標準出力を保存したい場合は、`bash scripts/mpirun/345m_dp16.sh > log.txt` などとしてください。

### mpirun: カスタマイズ引数付

`345m_custom.sh` スクリプトを使うことにより、ノード数、ノードあたりGPU数、ホストファイルをコマンドラインから指定できるようになります。（その他、パイプライン並列、テンソル並列の引数も用意されていますが、現状では反映されず、将来的に利用可能になります）


```bash
bash scripts/mpirun/345m_custom.sh -h
Usage: scripts/mpirun/345m_custom.sh [-n|--nodes <number of nodes>] [-g|--gpus <number of GPUs per node>] [-f|--hostfile <hostfile path>] [--pp <Pipeline parallel size>] [--tp <Tensor parallel size>]
```

下の例では、１ノードあたり８GPUで８ノード実行（合計64GPU）し、ホストファイルは `hostfile` パイプライン並列、テンソル並列なし（並列数: 1）で実行します。

```bash
source .env/bin/activate
bash scripts/mpirun/345m_custom.sh -n 8 -g 8 -f hostfile --pp 1 --tp 1
```


## Appendix

### hostfile (推奨)

上記の `mpirun` コマンドの `-H` オプションでノードの名前を並べてスクリプトに記述する代わりに、ノードの名前を `hostfile` に記述して、`-hostfile` オプションで読み込ませることもできます。

```text
10.2.72.135 slots=8
10.2.72.136 slots=8
```

```bash
mpirun -np $WORLD_SIZE --npernode $GPUS_PER_NODE \
  -hostfile hostfile \
  -x MASTER_ADDR=10.2.72.135 \
```

### 実行前計算ノードチェック

マルチノード実行前に、各計算ノードに正常にGPUデバイスが認識されているか、numactl がインストールされているかどうか、hostfile と `check_hosts.sh` スクリプトでチェックすることができます。
実行例:

```bash
bash scripts/check_hosts.sh hostfile

Host 10.2.72.135: OK: 8 GPUs found
Host 10.2.72.135: OK: numactl exists
Host 10.2.72.136: OK: 8 GPUs found
Host 10.2.72.136: OK: numactl exists
```

もしGPUが部分的に認識されていない場合、該当するノードにssh でログインし、MIG を無効化してみてください。例えば、ノード `10.2.72.136` で GPU ID 2 が認識されない場合、以下のコマンドを実行してみてください（要 sudo）

```bash
ssh 10.2.72.136
sudo nvidia-smi -i 2 -mig 0
```

また、新しくノードをデプロイしたままでは numactl がインストールされていないため、 `Host 10.2.72.136: Warning: numactl not found. Please install if necessary` と警告が出る場合がありますが、これは conda 環境でのみ必要であり、推奨されている pyenv + env では不要なので無視してください。


### python 環境設定

pyenv を install する前に以下のようなコマンドを打ち

```bash
export PYENV_ROOT="model/<user-dir>/.pyenv"
```

bashrcに書き込むものも、上記のパスに合わせれば `~/`以下でないところにpyenvをinstallできます。

また python cache も

```bash
# pip cache
export PIP_CACHE_DIR="<user-dir>/.cache/pip"
```

とすることで、cache作成先を変えることができます。

