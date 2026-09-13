#!/usr/bin/env node
/**
 * 把 dist/ 发布到 GitHub Pages（gh-pages 分支）。
 *
 *   pnpm deploy:pages              # 构建后发到 origin 的 gh-pages 分支
 *   pnpm deploy:pages --dry-run    # 只构建，不推送
 *
 * ⚠️ GitHub Pages **配不了 COOP/COEP 响应头**，所以线上跑的是「未跨源隔离」的路径：
 * SharedArrayBuffer 不可用，MediaPipe 用单线程 WASM、ORT 的 numThreads 退到 1。
 * 功能完整，只是比内网自托管慢一些。这是已知且刻意接受的降级，不是 bug。
 *
 * 内网自托管请直接用 `pnpm build` 的产物，并参考 README 里的 nginx 片段把
 * COOP/COEP 配上，那才是性能最好的一档。
 */
import { execFileSync } from 'node:child_process'
import { existsSync, rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const dryRun = process.argv.includes('--dry-run')

function run(cmd, args, opts = {}) {
  return execFileSync(cmd, args, { cwd: root, stdio: 'inherit', ...opts })
}

function git(args, opts) {
  return execFileSync('git', args, { cwd: root, encoding: 'utf8', ...opts }).trim()
}

// 1) 构建。build 里已经带了 tsc --noEmit，类型不过就不出产物。
console.log('▶ 构建')
run('pnpm', ['build'])

if (!existsSync(resolve(root, 'dist/index.html'))) {
  console.error('✗ dist/index.html 不存在，构建没有产出')
  process.exit(1)
}

if (dryRun) {
  console.log('✓ dry-run：dist/ 已就绪，未推送')
  process.exit(0)
}

// 2) 确认这是个带远端的 git 仓库，别默默把东西推错地方
try {
  git(['rev-parse', '--is-inside-work-tree'])
} catch {
  console.error('✗ 当前目录不是 git 仓库。先 `git init` 并配上远端再发布。')
  process.exit(1)
}
if (!git(['remote']).split('\n').filter(Boolean).length) {
  console.error('✗ 没有配置任何 git remote，无法发布。')
  process.exit(1)
}

// 3) 用 worktree 把 gh-pages 分支挂起来，把 dist 内容倒进去再提交。
//    用 worktree 而不是切分支，是为了不动当前工作区（构建产物和源码都可能还在改）。
const WT = '.deploy-gh-pages'
rmSync(resolve(root, WT), { recursive: true, force: true })

const hasBranch = (() => {
  try {
    git(['rev-parse', '--verify', '--quiet', `refs/heads/gh-pages`])
    return true
  } catch {
    return false
  }
})()

if (hasBranch) {
  run('git', ['worktree', 'add', WT, 'gh-pages'])
} else {
  run('git', ['worktree', 'add', '--orphan', '-b', 'gh-pages', WT])
}

try {
  rmSync(resolve(root, WT), { recursive: true, force: true })
  run('cp', ['-r', 'dist/.', `${WT}/`])
  // .nojekyll：不然 Pages 会拿 Jekyll 处理一遍，带下划线的目录会被吃掉
  run('touch', [`${WT}/.nojekyll`])

  const staged = execFileSync('git', ['status', '--porcelain'], {
    cwd: resolve(root, WT),
    encoding: 'utf8',
  }).trim()
  if (!staged) {
    console.log('✓ 产物没有变化，无需发布')
  } else {
    run('git', ['add', '-A'], { cwd: resolve(root, WT) })
    run('git', ['commit', '-m', `deploy: ${new Date().toISOString()}`], { cwd: resolve(root, WT) })
    run('git', ['push', 'origin', 'gh-pages'], { cwd: resolve(root, WT) })
    console.log('✓ 已推送到 origin/gh-pages')
  }
} finally {
  try {
    run('git', ['worktree', 'remove', '--force', WT])
  } catch {
    rmSync(resolve(root, WT), { recursive: true, force: true })
  }
}
