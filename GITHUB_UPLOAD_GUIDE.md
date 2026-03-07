# GitHub 上传指南

## 步骤 1: 创建 GitHub 仓库

### 方法一：通过 GitHub 网页

1. 登录 GitHub (https://github.com)
2. 点击右上角 "+" 号 → "New repository"
3. 填写信息：
   - Repository name: `pop-pdhg-reproducibility`
   - Description: `Minimum reproducible package for Pop-PDHG: Quantum-inspired population-based primal-dual hybrid gradient for mixed integer programming`
   - Visibility: Public (推荐) 或 Private
   - 勾选 "Add a README file" (可选，我们已有 README.md)
4. 点击 "Create repository"

### 方法二：通过 GitHub CLI

```bash
# 安装 GitHub CLI (如果未安装)
# macOS: brew install gh
# Ubuntu: sudo apt install gh

# 登录
ght auth login

# 创建仓库
gh repo create pop-pdhg-reproducibility \
  --public \
  --description "Minimum reproducible package for Pop-PDHG" \
  --source=. \
  --remote=origin
```

## 步骤 2: 初始化 Git 仓库

在 reproducibility_package 目录中执行：

```bash
cd /Users/cjx-mbp/Documents/Quantum_Solver/reproducibility_package

# 初始化 git 仓库
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: Pop-PDHG reproducibility package

- Core Pop-PDHG implementation
- SHADE comparison baseline
- Example scripts for p0282 instance
- Complete documentation"

# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/pop-pdhg-reproducibility.git

# 推送
git branch -M main
git push -u origin main
```

## 步骤 3: 验证上传

1. 访问 `https://github.com/YOUR_USERNAME/pop-pdhg-reproducibility`
2. 检查所有文件是否已上传
3. 确认 README.md 正确显示

## 步骤 4: 创建 Release（可选）

### 通过 GitHub 网页

1. 在仓库页面点击右侧 "Releases"
2. 点击 "Create a new release"
3. 填写信息：
   - Tag version: `v1.0.0`
   - Release title: `Pop-PDHG Reproducibility Package v1.0`
   - Description: 复制 README.md 的内容
4. 点击 "Publish release"

### 通过 GitHub CLI

```bash
gh release create v1.0.0 \
  --title "Pop-PDHG Reproducibility Package v1.0" \
  --notes "Minimum reproducible code for the paper 'Quantum Tunneling for Discrete Optimization'"
```

## 步骤 5: 测试复现

让其他人测试复现：

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/pop-pdhg-reproducibility.git
cd pop-pdhg-reproducibility

# 安装依赖
pip install -r requirements.txt

# 下载数据
python data/download_data.py

# 运行示例
python examples/run_p0282.py
```

## 常见问题

### 1. 文件过大无法上传

如果数据文件太大：

```bash
# 使用 Git LFS (Large File Storage)
git lfs install
git lfs track "data/*.mps"
git add .gitattributes
git commit -m "Add Git LFS for large data files"
```

### 2. 包含敏感信息

如果意外上传了敏感信息：

```bash
# 从提交历史中删除文件
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/sensitive/file' \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送
git push origin --force --all
```

### 3. Gurobi license 问题

不要上传 Gurobi license 文件！确保 `.gitignore` 包含：

```
gurobi.lic
*.lic
```

## 提交记录建议

良好的提交记录有助于复现：

```bash
# 初始提交
git commit -m "Initial commit: core implementation"

# 添加功能
git commit -m "Add SHADE comparison baseline

- Implement SHADE differential evolution
- Add constraint handling for MIP
- Include p0282 test case"

# 修复问题
git commit -m "Fix: constraint violation calculation

- Correct max violation computation
- Update tests accordingly"

# 更新文档
git commit -m "Update README with citation info"
```

## 启用 GitHub Actions（可选）

创建 `.github/workflows/test.yml`：

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run tests
      run: |
        python -m pytest tests/
```

## 获取 DOI（推荐）

通过 Zenodo 获取 DOI：

1. 登录 Zenodo (https://zenodo.org)
2. 关联 GitHub 账户
3. 选择仓库并启用
4. 创建新 release 时自动获取 DOI

这会在论文中引用：

```bibtex
@software{pop_pdhg_2024,
  author = {Authors},
  title = {Pop-PDHG Reproducibility Package},
  year = {2024},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## 联系与支持

如有问题，请在 GitHub 仓库创建 Issue。
