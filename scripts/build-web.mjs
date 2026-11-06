import { build } from 'esbuild';
import { copyFile, mkdir } from 'node:fs/promises';

await mkdir('src/aviutl_whisper/web/vendor', { recursive: true });
await build({
    entryPoints: ['src/aviutl_whisper/web/video-tools.js'],
    outfile: 'src/aviutl_whisper/web/vendor/video-tools.js',
    bundle: true, format: 'iife', globalName: 'VideoTools',
    target: 'chrome120', minify: true, legalComments: 'eof',
});
await copyFile('node_modules/mediabunny/LICENSE', 'src/aviutl_whisper/web/vendor/mediabunny-LICENSE.txt');
