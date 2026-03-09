# GitHub 更新与推送说明

当前最小复现仓库目录：

```bash
/Users/cjx-mbp/Documents/我的坚果云/科研/论文/2026_Quantum_Solver/Plan 1/legacy/reproducibility_package
```

当前远程仓库：

```bash
origin = https://github.com/IMUJason/pop-pdhg-reproducibility.git
```

## 日常更新流程

```bash
cd "/Users/cjx-mbp/Documents/我的坚果云/科研/论文/2026_Quantum_Solver/Plan 1/legacy/reproducibility_package"

git status
git add .
git commit -m "Describe the update briefly"
git push origin main
```

## 首次拉取

```bash
git clone https://github.com/IMUJason/pop-pdhg-reproducibility.git
cd pop-pdhg-reproducibility
```

## 环境安装

推荐使用 `uv`：

```bash
uv sync --extra dev
```

兼容旧方式：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

官方 MIPLIB 实例不随仓库上传。按需执行：

```bash
python data/prepare_repro_data.py --paper-suite
```

或使用 `uv`：

```bash
uv run python data/prepare_repro_data.py --paper-suite
```

## 复现检查

```bash
uv run pytest tests/test_pdhg.py -v
uv run python experiments/scripts/run_main_benchmark_robustness.py \
  --instances p0033 knapsack_50 \
  --seeds 11 17 \
  --max-iter 300 \
  --population-size 8
```

## 注意事项

- 不要上传官方 MIPLIB `.mps` / `.mps.gz` 文件。
- `data/miplib2017/knapsack_50.mps` 是仓库内保留的自定义小算例。
- LaTeX 编译产物、本地缓存、投稿辅助打包文件已由 `.gitignore` 排除。
