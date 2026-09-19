/**
 * 使用本机 Chrome 的 DevTools 协议验证真实页面，无需新增测试依赖。
 * 先运行 npm run build，再运行 node scripts/ui-smoke.mjs。
 * 所有 API 都被合成数据拦截，不读取真实患者或调用真实模型；这不是后端集成测试。
 * CHROME_PATH 可覆盖浏览器路径，截图写入已忽略的 .cache/ui-smoke。
 */
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, mkdtemp } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, join, extname, dirname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = join(root, '.cache/ui-smoke');
await mkdir(output, { recursive: true });
const profile = await mkdtemp(join(output, 'chrome-'));
const chrome = process.env.CHROME_PATH ?? [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome', '/usr/bin/chromium'
].find(existsSync);
assert.ok(chrome, 'Set CHROME_PATH to a local Chrome executable');
const server = createServer(async (request, response) => {
  const name = new URL(request.url, 'http://localhost').pathname;
  const file = resolve(root, 'dist/dentalai-copilot-web', `.${name === '/' ? '/index.html' : name}`);
  if (!file.startsWith(join(root, 'dist/dentalai-copilot-web') + sep)) {
    response.writeHead(403).end(); return;
  }
  try {
    const content = await readFile(file);
    response.setHeader('Content-Type', { '.js': 'application/javascript', '.css': 'text/css', '.html': 'text/html' }[extname(file)] ?? 'application/octet-stream');
    response.end(content);
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = spawn(chrome, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { windowsHide: true, stdio: 'ignore' });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
try {
  let port;
  for (let attempt = 0; attempt < 100; attempt++) {
    try { port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; } catch { await delay(100); }
  }
  assert.ok(port, 'Chrome did not expose a debugging port');
  const pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  socket = new WebSocket(pages.find(page => page.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let sequence = 0;
  const pending = new Map();
  const errors = [];
  let handleRequest = async () => {};
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.id) {
      const callback = pending.get(message.id);
      pending.delete(message.id);
      message.error ? callback?.reject(new Error(JSON.stringify(message.error))) : callback?.resolve(message.result);
    } else if (message.method === 'Fetch.requestPaused') {
      handleRequest(message.params).catch(error => errors.push(error.message));
    } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
  };
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 15000);
    pending.set(id, { resolve: value => { clearTimeout(timeout); resolve(value); }, reject: error => { clearTimeout(timeout); reject(error); } });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const until = async expression => {
    for (let i = 0; i < 100; i++) { if (await evaluate(expression)) return; await delay(50); }
    throw new Error(`Timed out: ${expression}`);
  };
  const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const screenshot = async name => {
    const { data } = await call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    await writeFile(join(output, `${name}.png`), Buffer.from(data, 'base64'));
  };
  // 两项独立动作验证顺序、默认拒绝，以及 approve/reject 混合决定。
  const review = { interrupt_id: 'demo-interrupt-1', actions: [
    { name: 'change_records', description: 'Create a synthetic appointment.', arguments: { operation: 'create_appointment', patient_id: 'demo-patient', reason: 'Demo only' } },
    { name: 'change_records', description: 'Add a synthetic note.', arguments: { operation: 'add_note', patient_id: 'demo-patient', content: 'Synthetic note only' } }
  ] };
  const submissions = [];
  let authenticated = true;
  let rejectResume = false;
  const longAnswer = {
    answer: 'Synthetic patient information.\n'.repeat(60),
    evidence_ids: Array.from({ length: 20 }, (_, index) => `synthetic-evidence-${index}`),
    limitations: Array.from({ length: 25 }, () => 'Synthetic limitation for layout regression. '.repeat(12))
  };
  handleRequest = async ({ requestId, request }) => {
    const path = new URL(request.url).pathname;
    let body = {}; let status = 200; let type = 'application/json';
    if (request.method === 'OPTIONS') status = 204;
    else if (path.endsWith('/auth/me')) { body = { id: 'demo-user', username: 'demo', display_name: 'Demo User' }; if (!authenticated) status = 401; }
    else if (path.endsWith('/patients')) body = [{ id: 'demo-patient', name: 'Demo Patient', date_of_birth: '1990-01-01', patient_number: 'DEMO-001', medical_record_number: 'MR-DEMO', dox_patient_id: null }];
    else if (path.endsWith('/appointments')) body = [];
    else if (path.endsWith('/chat/connect')) body = { connected: true, mock: true, provider: 'mock' };
    else if (path.endsWith('/conversations')) body = request.method === 'POST' ? { id: 'demo-thread', patient_ids: ['demo-patient'] } : [{ id: 'demo-thread', patient_ids: ['demo-patient'], created_at: '2026-09-18T09:00:00Z' }];
    else if (path.endsWith('/demo-thread')) body = { id: 'demo-thread', messages: [], review, can_continue: false };
    else if (path.endsWith('/messages')) { type = 'text/event-stream'; body = `data: ${JSON.stringify({ type: 'approval', review, message: 'Awaiting review' })}\n\n`; }
    else if (path.endsWith('/resume')) {
      submissions.push(JSON.parse(request.postData));
      await delay(150);
      type = 'text/event-stream';
      if (rejectResume) { body = `data: ${JSON.stringify({ type: 'error', message: 'Approval is stale or decisions do not match the pending actions' })}\n\n`; }
      else { body = `data: ${JSON.stringify({ type: 'result', result: longAnswer, mock: true })}\n\n`; }
    }
    else { throw new Error(`Unexpected API: ${request.method} ${path}`); }
    await call('Fetch.fulfillRequest', { requestId, responseCode: status, responseHeaders: [
      { name: 'Content-Type', value: type }, { name: 'Access-Control-Allow-Origin', value: origin },
      { name: 'Access-Control-Allow-Credentials', value: 'true' }, { name: 'Access-Control-Allow-Headers', value: 'content-type' },
      { name: 'Access-Control-Allow-Methods', value: 'GET,POST,OPTIONS' }
    ], body: Buffer.from(typeof body === 'string' ? body : JSON.stringify(body)).toString('base64') });
  };
  console.log('Chrome connected; loading fixture-backed application.');
  await call('Runtime.enable'); await call('Page.enable');
  await call('Fetch.enable', { patterns: [{ urlPattern: 'http://localhost:8000/*' }] });
  await call('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await call('Page.navigate', { url: origin });
  await until(`document.querySelector('.patient') !== null`);
  await evaluate(`{ const search = document.querySelector('.search-field input'); search.value = 'no-such-patient'; search.dispatchEvent(new Event('input', { bubbles: true })); }`);
  await until(`document.querySelector('.patient') === null`);
  await evaluate(`{ const search = document.querySelector('.search-field input'); search.value = ''; search.dispatchEvent(new Event('input', { bubbles: true })); }`);
  await until(`document.querySelector('.patient') !== null`);
  await screenshot('workbench-desktop');
  await click('.model-connect');
  await until(`document.querySelector('.model-connect').textContent.includes('Mock')`);
  await click('.select-patient input');
  await click('.suggestions button');
  await click('.composer button');
  await until(`document.querySelector('dialog').open && !document.querySelector('.modal-footer .primary').disabled`);
  assert.equal(await evaluate(`document.querySelectorAll('dialog input[value="reject"]:checked').length`), 2);
  assert.equal(await evaluate(`document.querySelector('dialog').matches(':modal')`), true);
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('dialog pre'), node => JSON.parse(node.textContent))`), review.actions.map(action => action.arguments));
  await screenshot('approval-desktop');
  await call('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await call('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await until(`!document.querySelector('dialog').open`);
  assert.equal(submissions.length, 0, 'Dismissing must not submit decisions');
  await click('.review-banner button');
  await until(`document.querySelector('dialog').open`);
  await click('dialog input[value="approve"]');
  await click('.modal-footer .primary');
  await until(`!document.querySelector('dialog').open`);
  assert.deepEqual(submissions[0], { interrupt_id: review.interrupt_id, decisions: [{ type: 'approve' }, { type: 'reject' }] });
  // 长答案与大量说明全部进入同一滚动区，展开后也不能压缩输入框或对话窗口。
  await until(`document.querySelector('.answer-details') !== null`);
  assert.equal(await evaluate(`document.querySelector('.answer-details').open`), false);
  const contentHeight = await evaluate(`document.querySelector('.chat-content').clientHeight`);
  await click('.answer-details summary');
  for (const width of [1440, 390, 768]) {
    await call('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: false });
    await delay(100);
    const geometry = await evaluate(`(() => { const area = document.querySelector('.chat-content'); const composer = document.querySelector('.composer'); return { height: area.clientHeight, scroll: area.scrollHeight, composer: composer.clientHeight, input: composer.querySelector('textarea').clientHeight }; })()`);
    assert.ok(geometry.height >= 280, `Chat collapsed at ${width}: ${JSON.stringify(geometry)}`);
    assert.ok(geometry.scroll > geometry.height);
    assert.ok(geometry.composer >= 120 && geometry.input >= 70);
    await evaluate(`document.querySelector('.chat-panel').scrollIntoView({ block: 'start' })`);
    await screenshot(`long-answer-${width}`);
  }
  await call('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  assert.equal(await evaluate(`document.querySelector('.chat-content').clientHeight`), contentHeight);
  // 恢复历史会话仍自动打开审批；真实后端使用 SSE error 表达过期审批，保留弹窗提示。
  await evaluate(`const history = document.querySelector('.history-select select'); history.value = 'demo-thread'; history.dispatchEvent(new Event('change', { bubbles: true }));`);
  await until(`document.querySelector('dialog').open`);
  rejectResume = true;
  await click('.modal-footer .primary');
  await until(`document.querySelector('.modal-footer .error')?.textContent.includes('stale')`);
  assert.equal(await evaluate(`document.querySelector('dialog').open`), true);
  assert.deepEqual(submissions[1].decisions, [{ type: 'reject' }, { type: 'reject' }]);
  await click('.modal-heading button');
  for (const width of [390, 768]) {
    await call('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: false });
    for (let section = 1; section <= 3; section++) {
      await click(`nav button:nth-child(${section})`); await delay(80);
      assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, `Overflow at ${width}, section ${section}`);
    }
    await screenshot(`imports-${width}`);
  }
  await click('nav button'); await click('.review-banner button');
  await until(`document.querySelector('dialog').open`);
  await call('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: false });
  await screenshot('approval-mobile');
  await click('.modal-heading button');
  await screenshot('workbench-mobile');
  authenticated = false;
  await call('Page.reload'); await until(`document.querySelector('.auth-card') !== null`);
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
  await screenshot('login-mobile');
  await call('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await screenshot('login-desktop');
  assert.deepEqual(errors, []);
  console.log('PASS: SSE approval/error, restore, mixed decisions, long answers/limitations, responsive layouts and login.');
  console.log(`Screenshots: ${output}`);
  await call('Browser.close');
} finally {
  socket?.close(); browser.kill(); server.close();
}
