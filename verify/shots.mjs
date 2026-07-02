// Load the real page in headless Chromium, run each proof's training, and
// screenshot the results so we can confirm the page renders and the money shots
// actually draw. Assumes a static server is running at http://localhost:8842.
import puppeteer from 'puppeteer';

const BASE = process.env.BASE || 'http://localhost:8842';
const OUT = process.env.OUT || '/private/tmp/claude-501/-Users-user-projects-Madhulatha-Sandeep-learning-ws/8600fb57-6c7d-45ca-99ab-ada654b13a3e/scratchpad';

const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 2 });

const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));

await page.goto(BASE + '/index.html', { waitUntil: 'networkidle0', timeout: 60000 });
await new Promise((r) => setTimeout(r, 1500));

// hero
await page.screenshot({ path: `${OUT}/shot-hero.png` });

async function clickAndWait(sectionSel, btnSel, ms) {
  await page.$eval(sectionSel, (el) => el.scrollIntoView());
  await new Promise((r) => setTimeout(r, 400));
  await page.click(btnSel);
  // wait for training to finish: button re-enables (text back to "Retrain")
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    const disabled = await page.$eval(btnSel, (b) => b.disabled);
    if (!disabled) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  await new Promise((r) => setTimeout(r, 600));
}

await clickAndWait('#proof-1', '#p1-train', 60000);
await page.$eval('#proof-1', (el) => el.scrollIntoView());
await new Promise((r) => setTimeout(r, 300));
await page.screenshot({ path: `${OUT}/shot-p1.png` });
console.log('p1 status:', await page.$eval('#p1-status', (e) => e.textContent));
console.log('p1 acc linear/relu:', await page.$eval('#p1-linear-acc', (e) => e.textContent), '/', await page.$eval('#p1-relu-acc', (e) => e.textContent));

await clickAndWait('#proof-2', '#p2-train', 90000);
await page.$eval('#proof-2', (el) => el.scrollIntoView());
await new Promise((r) => setTimeout(r, 300));
await page.screenshot({ path: `${OUT}/shot-p2.png` });
console.log('p2 acc 1/5/relu:', await page.$eval('#p2-one-acc', (e) => e.textContent), '/', await page.$eval('#p2-five-acc', (e) => e.textContent), '/', await page.$eval('#p2-relu-acc', (e) => e.textContent));
console.log('p2 matrix:', (await page.$eval('#p2-matrix', (e) => e.textContent)).replace(/\n/g, ' | '));

await clickAndWait('#proof-3', '#p3-train', 90000);
await page.$eval('#proof-3', (el) => el.scrollIntoView());
await new Promise((r) => setTimeout(r, 300));
await page.screenshot({ path: `${OUT}/shot-p3.png` });
console.log('p3 nn:', await page.$eval('#p3-nn', (e) => e.textContent));

await clickAndWait('#proof-4', '#p4-train', 120000);
await page.$eval('#proof-4', (el) => el.scrollIntoView());
await new Promise((r) => setTimeout(r, 300));
await page.screenshot({ path: `${OUT}/shot-p4.png` });
console.log('p4 gap:', await page.$eval('#p4-gap', (e) => e.textContent));

console.log('\nrender errors:', errors.length ? errors : 'none');
await browser.close();
