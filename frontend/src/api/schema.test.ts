// @vitest-environment node
import { readFileSync } from 'node:fs';
import openapiTS, { astToString } from 'openapi-typescript';
import { describe, expect, it } from 'vitest';

const strip = (s: string) =>
  s
    .replace(/^\/\*\*[\s\S]*?\*\/\s*/, '')
    .replace(/\r\n/g, '\n')
    .trim();

describe('generated API types', () => {
  it('src/api/schema.d.ts is up to date with api/openapi.yaml (run `npm run gen:api`)', async () => {
    const ast = await openapiTS(new URL('../../../api/openapi.yaml', import.meta.url));
    const onDisk = readFileSync(new URL('./schema.d.ts', import.meta.url), 'utf8');
    expect(strip(astToString(ast))).toBe(strip(onDisk));
  });
});
