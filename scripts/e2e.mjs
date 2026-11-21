// 作品页 e2e：真浏览器打开 site/，量版式、验资源、截图。
// 用法：node scripts/e2e.mjs [--url http://127.0.0.1:8611/]
import { chromium } from 'playwright-core';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SITE = path.join(ROOT, 'site');
const SHOTS = path.join(ROOT, 'out', 'shots');
const PORT = 8611;

const PASS = [];
const FAIL = [];
const ok = (name, cond, detail = '') => (cond ? PASS : FAIL).push(`${name}${detail ? ' — ' + detail : ''}`);

function serve() {
  const p = spawn('py', ['-3', '-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], {
    cwd: SITE, stdio: 'ignore',
  });
  return p;
}

async function main() {
  mkdirSync(SHOTS, { recursive: true });
  const server = serve();
  await new Promise((r) => setTimeout(r, 1200));
  const exe = process.env.PW_CHROME ||
    path.join(process.env.USERPROFILE || '', 'AppData', 'Local', 'ms-playwright',
              'chromium-1248', 'chrome-win64', 'chrome.exe');
  const browser = await chromium.launch({ executablePath: exe });
  const errors = [];

  for (const [label, vp, dsf] of [
    ['desktop', { width: 1440, height: 900 }, 1],
    ['mobile', { width: 390, height: 844 }, 2],
  ]) {
    const ctx = await browser.newContext({ viewport: vp, deviceScaleFactor: dsf });
    const page = await ctx.newPage();
    page.on('console', (m) => { if (m.type() === 'error') errors.push(`${label}: ${m.text()}`); });
    page.on('pageerror', (e) => errors.push(`${label}: ${e.message}`));
    await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: 'load' });
    await page.waitForTimeout(900);

    // 字体真的加载（子集里有这几个字）
    const fontOk = await page.evaluate(async () => {
      await document.fonts.ready;
      return document.fonts.check('16px zw', '第五个轮子茅箭');
    });
    ok(`${label}: 子集中文字体已加载`, fontOk);

    // 标题与段落
    const h1 = await page.textContent('h1');
    ok(`${label}: 标题是「第五个轮子」`, h1.replace(/\s/g, '') === '第五个轮子', h1);

    // 视频元素：metadata 能读到时长
    const meta = await page.evaluate(async () => {
      const v = document.querySelector('video');
      if (!v) return null;
      if (v.readyState < 1) {
        await new Promise((res) => {
          const t = setTimeout(res, 6000);
          v.addEventListener('loadedmetadata', () => { clearTimeout(t); res(); }, { once: true });
          v.load();
        });
      }
      return { d: v.duration, w: v.videoWidth, h: v.videoHeight, err: v.error?.code ?? null };
    });
    ok(`${label}: 视频可读 metadata`, !!meta && meta.d === meta.d, JSON.stringify(meta));
    if (meta && meta.d === meta.d) {
      ok(`${label}: 时长 156 s`, Math.abs(meta.d - 156) < 0.5, `${meta.d}`);
      ok(`${label}: 1920×1080`, meta.w === 1920 && meta.h === 1080, `${meta.w}×${meta.h}`);
    }

    // 六帧都真加载
    const imgs = await page.evaluate(() =>
      [...document.querySelectorAll('.grid img')].map((i) => ({ src: i.getAttribute('src'), w: i.naturalWidth })));
    ok(`${label}: 六帧全部加载`, imgs.length === 6 && imgs.every((i) => i.w > 0),
      JSON.stringify(imgs.map((i) => i.w)));

    // 分镜表 19 行
    const rows = await page.evaluate(() => document.querySelectorAll('table tbody tr').length);
    ok(`${label}: 表格行数 = 19 + 10`, rows === 29, `${rows}`);

    // 没有横向溢出
    const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    ok(`${label}: 无横向溢出`, over <= 1, `${over}px`);

    // 正文可读性：正文字号 ≥ 14，行高 ≥ 1.5
    const typo = await page.evaluate(() => {
      const s = getComputedStyle(document.body);
      return { fs: parseFloat(s.fontSize), lh: parseFloat(s.lineHeight) / parseFloat(s.fontSize) };
    });
    ok(`${label}: 正文 ≥14px / 行高 ≥1.5`, typo.fs >= 14 && typo.lh >= 1.5, JSON.stringify(typo));

    // 关键块不重叠：标题块与 meta 行
    const boxes = await page.evaluate(() => {
      const r = (sel) => { const e = document.querySelector(sel); const b = e.getBoundingClientRect(); return { t: b.top, b: b.bottom, l: b.left, r: b.right }; };
      return { h1: r('h1'), sub: r('.sub'), meta: r('.meta') };
    });
    ok(`${label}: 标题/副题/meta 不重叠`,
      boxes.h1.b <= boxes.sub.t + 1 && boxes.sub.b <= boxes.meta.t + 1, JSON.stringify(boxes));

    await page.screenshot({ path: path.join(SHOTS, `page-${label}.png`), fullPage: true });
    await ctx.close();
  }

  ok('无 JS 报错', errors.length === 0, errors.slice(0, 3).join(' | '));
  await browser.close();
  server.kill();

  console.log(`\nPASS ${PASS.length} / FAIL ${FAIL.length}`);
  FAIL.forEach((f) => console.log('  ✗', f));
  if (!FAIL.length) console.log('  全部通过');
  process.exit(FAIL.length ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(2); });
