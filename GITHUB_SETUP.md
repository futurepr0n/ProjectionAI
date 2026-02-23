# GitHub Setup Instructions

## Option 1: Create Repository on GitHub (Recommended)

1. **Go to GitHub**: https://github.com/new
2. **Repository name**: `ProjectionAI`
3. **Description**: `MLB Betting Projection System using XGBoost`
4. **Visibility**: Private (recommended) or Public
5. **Don't initialize**: Leave "Add a README file" unchecked (we already have one)
6. **Click "Create repository"**

Then push your code:

```bash
cd ~/Development/ProjectionAI

# Rename branch to main
git branch -M main

# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ProjectionAI.git

# Push to GitHub
git push -u origin main
```

## Option 2: Using GitHub CLI (if you install it)

```bash
# Install GitHub CLI (Ubuntu/Debian)
sudo apt update
sudo apt install gh

# Authenticate
gh auth login

# Create repository and push
cd ~/Development/ProjectionAI
gh repo create ProjectionAI --public --source=. --remote=origin
git push -u origin main
```

## Option 3: Use Personal Access Token

1. **Generate token**:
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Scopes: `repo` (full control of private repositories)
   - Generate and copy the token

2. **Push with token**:

```bash
cd ~/Development/ProjectionAI

# Add remote with token (replace YOUR_USERNAME and YOUR_TOKEN)
git remote add origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/ProjectionAI.git

# Push
git push -u origin main
```

## Current Status

✅ Git initialized
✅ Initial commit created (47 files, 11,074 lines)
✅ Branch renamed to main
❌ No GitHub remote configured yet

## Files Committed

- Core ML models (matchup_model_v3.py)
- Results dashboard (generate_report.py, start_server.py)
- Data processing scripts
- Documentation (README.md, PROJECT_PLAN.md)
- Configuration files
- Memory and analysis logs

## Next Steps

1. Create GitHub repository
2. Add remote with: `git remote add origin <url>`
3. Push with: `git push -u origin main`

---

**Remember**: Your database credentials (in data/config.py) contain sensitive information. These are in .gitignore and won't be pushed to GitHub.
