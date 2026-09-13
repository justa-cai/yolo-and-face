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
import { existsSync, mkdirSync, rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const dryRun = process.argv.includes('--dry-run')

/** 跑子进程。opts.cwd 相对 root，这里统一转成绝对路径。 */
function run(cmd, args, opts = {}) {
  const { cwd, ...rest } = opts
  return execFileSync(cmd, args, {
    cwd: cwd ? resolve(root, cwd) : root,
    stdio: 'inherit',
    ...rest,
  })
}

function git(args, opts = {}) {
  return execFileSync('git', args, {
    cwd: opts.cwd ? resolve(root, opts.cwd) : root,
    encoding: 'utf8',
  }).trim()
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

// 3) 把 dist 倒进一个一次性仓库，再强推成 origin/gh-pages。
//
//    刻意不用 `git worktree add --orphan`：那需要 Git 2.42+，而本机是 2.34，
//    直接报 unknown option。这里改成「新建一个临时仓库 -> 提交 -> 强推」，
//    对 Git 版本没有要求，也不动当前工作区。
//
//    每次都强推（单提交历史）而不是追加：产物每次几乎全变，追加会让 gh-pages
//    的分支历史无限膨胀，强推让它稳定在 117MB 左右。
const WT = resolve(root, 'tmp/deploy-gh-pages')
rmSync(WT, { recursive: true, force: true })
mkdirSync(WT, { recursive: true })

const remoteUrl = git(['remote', 'get-url', 'origin'])

try {
  run('git', ['init', '-b', 'gh-pages'], { cwd: WT })
  run('git', ['remote', 'add', 'origin', remoteUrl], { cwd: WT })

  // dist 里没有 .git，直接倒进来不会碰到上面刚建好的仓库
  run('cp', ['-r', 'dist/.', `${WT}/`])
  // .nojekyll：不然 Pages 会拿 Jekyll 处理一遍，带下划线的目录会被吃掉
  run('touch', [`${WT}/.nojekyll`])

  run('git', ['add', '-A'], { cwd: WT })
  run('git', ['commit', '-q', '-m', `deploy: ${new Date().toISOString()}`], { cwd: WT })
  run('git', ['push', '--force', 'origin', 'gh-pages'], { cwd: WT })
  console.log('✓ 已强推到 origin/gh-pages')
} finally {
  rmSync(WT, { recursive: true, force: true })
}
