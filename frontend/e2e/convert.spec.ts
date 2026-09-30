import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

const SAMPLE = fileURLToPath(new URL('../../samples/01_logo_4color.png', import.meta.url));
const STAGE_TEXT =
  /Queued|Uploading|Preprocessing|Classifying image|Quantizing colors|Extracting lines|Tracing vectors|Assembling SVG|Exporting files|Evaluating quality|Done/;

async function expectNoHorizontalScroll(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow, 'page must not scroll horizontally').toBeLessThanOrEqual(0);
}

async function openApp(page: Page) {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1, name: 'VectorForge' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Choose image' })).toBeEnabled();
}

test('upload -> progress -> result -> palette edit + re-run -> download', async ({ page }) => {
  await openApp(page);
  await expectNoHorizontalScroll(page);

  // Upload through the file picker.
  await page.getByTestId('file-input').setInputFiles(SAMPLE);
  await expect(page.getByTestId('selected-file')).toContainText('01_logo_4color.png');

  // Settings panel is labelled and operable.
  await page.getByLabel('Mode').selectOption('flat_color');
  await page.getByLabel('Mode').selectOption('auto');
  await page.getByRole('checkbox', { name: 'EPS' }).uncheck();
  await page.getByRole('checkbox', { name: 'EPS' }).check();
  await expect(page.getByRole('checkbox', { name: 'SVG (always)' })).toBeDisabled();
  await expect(page.getByRole('checkbox', { name: 'SVG (always)' })).toBeChecked();

  // Convert -> progress with a stage name.
  await page.getByRole('button', { name: 'Convert' }).click();
  const progress = page.getByTestId('progress');
  await expect(progress).toBeVisible();
  await expect(progress.getByRole('progressbar', { name: 'Conversion progress' })).toBeVisible();
  await expect(page.getByTestId('progress-stage')).toHaveText(STAGE_TEXT);

  // Result.
  const result = page.getByTestId('result');
  await expect(result).toBeVisible();
  await expect(progress).toBeHidden();
  await expect(page.getByTestId('image-class')).toContainText('Flat color');
  await expect(page.getByTestId('quality-badge')).toHaveText(/All 6 quality checks passed/);
  await expect(page.getByTestId('check-ssim')).toContainText('Pass');
  await expect(page.getByTestId('palette-entry')).toHaveCount(4);

  // The SVG is rendered through <img> (never inlined) and actually loads.
  const vectorImg = page.getByAltText('Vector result (SVG)');
  await expect(vectorImg).toHaveJSProperty('naturalWidth', 512);
  expect(await page.locator('main svg[viewBox="0 0 512 512"]').count()).toBe(0);

  // Synchronized zoom: keyboard on the overlay, then side-by-side panes share the view.
  await page.getByTestId('pane-overlay').focus();
  await page.keyboard.press('+');
  await expect(page.getByTestId('zoom-level')).toHaveText('125%');
  await page.getByRole('button', { name: 'Zoom in' }).click();
  await expect(page.getByTestId('zoom-level')).toHaveText('156%');
  await page.getByRole('button', { name: 'Side by side' }).click();
  const transforms = await page
    .locator('[data-testid="pane-original"] > div, [data-testid="pane-vector"] > div')
    .evaluateAll((els) => els.map((el) => (el as HTMLElement).style.transform));
  expect(transforms).toHaveLength(2);
  expect(transforms[0]).toContain('scale(1.5625)');
  expect(transforms[0]).toBe(transforms[1]);
  await page.getByRole('button', { name: 'Reset view' }).click();
  await expect(page.getByTestId('zoom-level')).toHaveText('100%');
  await expectNoHorizontalScroll(page);

  // Palette: edit color 2, merge colors 3 + 4, then re-run.
  const hex2 = page.getByRole('textbox', { name: 'Color 2 hex value' });
  await hex2.fill('#B22222');
  await hex2.blur();
  await expect(hex2).toHaveValue('#b22222');
  await page.getByRole('checkbox', { name: /Select color 3/ }).check();
  await page.getByRole('checkbox', { name: /Select color 4/ }).check();
  await page.getByRole('button', { name: 'Merge selected (2)' }).click();
  await expect(page.getByTestId('palette-entry')).toHaveCount(3);
  await page.getByRole('button', { name: 'Re-run with edited palette' }).click();

  await expect(page.getByTestId('progress')).toBeVisible();
  await expect(page.getByTestId('rerun-note')).toContainText('custom palette');
  await expect(page.getByTestId('progress')).toBeHidden();
  await expect(page.getByTestId('palette-entry')).toHaveCount(3);
  await expect(page.getByRole('textbox', { name: 'Color 2 hex value' })).toHaveValue('#b22222');
  // Merging blue into yellow is a big color error: the badge shows failure with text + icon.
  await expect(page.getByTestId('quality-badge')).toHaveText(/quality checks failed/);
  await expect(page.getByTestId('check-max_delta_e')).toContainText('Fail');
  await expect(page.getByRole('button', { name: 'Re-run with these settings' })).toBeVisible();

  // Download SVG (from result.files) and check it reflects the edited palette.
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: /Download SVG/ }).click(),
  ]);
  expect(download.suggestedFilename()).toBe('01_logo_4color.svg');
  const svgPath = await download.path();
  const svg = readFileSync(svgPath, 'utf-8');
  expect(svg).toContain('<svg');
  expect(svg).toContain('fill="#b22222"');
  expect(svg).not.toContain('fill="#fac81e"');

  for (const label of ['AI (Illustrator)', 'EPS', 'PNG preview']) {
    await expect(page.getByRole('button', { name: new RegExp(`Download ${label.replace(/[()]/g, '\\$&')}`) })).toBeVisible();
  }
  await expectNoHorizontalScroll(page);
});

test('error paths: rejected type, undecodable image (422), failed job', async ({ page }) => {
  await openApp(page);

  // Client-side rejection of a non PNG/JPG file (limits come from GET /config).
  await page.getByTestId('file-input').setInputFiles({
    name: 'animation.gif',
    mimeType: 'image/gif',
    buffer: Buffer.from('GIF89a'),
  });
  const banner = page.getByTestId('error-banner');
  await expect(banner).toContainText('File not accepted');
  await expect(banner).toContainText('PNG or JPG');
  await expect(page.getByRole('button', { name: 'Convert' })).toBeDisabled();
  await banner.getByRole('button', { name: 'Dismiss' }).click();
  await expect(banner).toBeHidden();

  // Server-side 422: the bytes are not a decodable image.
  await page.getByTestId('file-input').setInputFiles({
    name: 'broken.png',
    mimeType: 'image/png',
    buffer: Buffer.from('this is not really a png'),
  });
  await page.getByRole('button', { name: 'Convert' }).click();
  await expect(page.getByRole('alert')).toContainText('Image could not be read');
  await expect(page.getByRole('alert')).toContainText('HTTP 422');

  // A job that fails in the pipeline (mock fails filenames containing "fail").
  await page.getByTestId('file-input').setInputFiles({
    name: 'will_fail.png',
    mimeType: 'image/png',
    buffer: readFileSync(SAMPLE),
  });
  await page.getByRole('button', { name: 'Convert' }).click();
  await expect(page.getByTestId('progress')).toBeVisible();
  const failure = page.getByRole('alert');
  await expect(failure).toContainText('Conversion failed');
  await expect(failure).toContainText('stage: Tracing vectors');
  await expect(failure).toContainText('code: stage_failed');
  await expect(page.getByTestId('result')).toHaveCount(0);
  await expectNoHorizontalScroll(page);
});

test('keyboard only: choose settings and convert', async ({ page }) => {
  await openApp(page);
  await page.getByTestId('file-input').setInputFiles(SAMPLE);

  const auto = page.getByRole('checkbox', { name: 'Auto-detect number of colors' });
  await auto.focus();
  await page.keyboard.press('Space');
  await expect(auto).not.toBeChecked();
  const colors = page.getByRole('slider', { name: /Max colors/ });
  await colors.focus();
  await page.keyboard.press('ArrowRight');
  await expect(colors).toHaveValue('9');

  await page.getByRole('radio', { name: 'High' }).focus();
  await page.keyboard.press('Space');
  await expect(page.getByRole('radio', { name: 'High' })).toBeChecked();

  const convert = page.getByRole('button', { name: 'Convert' });
  await convert.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByTestId('result')).toBeVisible();
  await expect(page.getByTestId('quality-badge')).toBeVisible();
});
