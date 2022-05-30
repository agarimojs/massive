require('dotenv').config();
const cluster = require('cluster');
const fs = require('fs');
const path = require('path');
const os = require('os');
const zlib = require('zlib');
const { initProcessors, getProcessor } = require('@agarimo/languages');
const { NeuralBig: Neural } = require('@agarimo/neural');
const {
  loadMassive,
  applyInCorpus,
  cleanCorpus,
  augmentNgrams,
} = require('./helpers');
const neuralSettings = require('./neural-settings.json');

const numCPUs = os.cpus().length;
const DATASET_PATH = './data';
const MODELS_PATH = './models';
const allowedLocales = undefined;
const useAnnot = false;

async function trainAndValidate(locale) {
  let corpus = loadMassive(DATASET_PATH, locale);
  corpus.data.forEach((srcItem) => {
    const item = srcItem;
    item.srcUtterances = item.utterances;
    item.srcTests = item.tests;
    if (useAnnot) {
      item.utterances = item.annotUtterances;
      item.tests = item.annotTests;
    }
  });
  const processor = getProcessor(locale);
  applyInCorpus(corpus, (str) => processor(str).join(' '), [
    'utterances',
    'tests',
  ]);
  corpus = cleanCorpus(corpus);
  applyInCorpus(corpus, (str) => augmentNgrams(str), ['utterances', 'tests']);
  const neural = new Neural(neuralSettings);
  console.time(`train ${locale}`);
  neural.train(corpus);
  console.timeEnd(`train ${locale}`);
  const measure = neural.measure();
  const model = zlib.deflateSync(JSON.stringify(neural.toJSON()));
  fs.writeFileSync(
    path.join(MODELS_PATH, `${locale}_${useAnnot ? 'annot_' : ''}model.zjson`),
    model
  );
  const accuracy = (measure.good * 100) / measure.total;
  process.send({ type: 'result', locale, accuracy });
}

if (cluster.isMaster) {
  const locales =
    allowedLocales ||
    fs
      .readdirSync(DATASET_PATH)
      .filter((file) => file.endsWith('.jsonl'))
      .map((file) => file.replace('.jsonl', ''));
  const pending = [...locales];
  const workers = {};
  console.time('time');
  const results = [];
  for (let i = 0; i < numCPUs; i += 1) {
    const worker = cluster.fork();
    workers[worker.process.pid] = worker;
    console.log(`Worker ${worker.process.pid} started`);
    worker.on('message', (msg) => {
      if (msg.type === 'gettask') {
        if (pending.length === 0) {
          worker.send({ type: 'exit' });
        } else {
          const locale = pending.shift();
          worker.send({ type: 'trainandvalidate', locale });
        }
      } else if (msg.type === 'result') {
        results.push({ locale: msg.locale, accuracy: msg.accuracy.toFixed(1) });
      }
    });
  }
  cluster.on('exit', (worker) => {
    delete workers[worker.process.pid];
    console.log(`Worker ${worker.process.pid} died`);
    if (Object.keys(workers).length === 0) {
      results.sort((a, b) => (a.locale < b.locale ? -1 : 1));
      console.log(results);
      console.log(
        results.reduce(
          (prev, curr) => prev + parseFloat(curr.accuracy, 10),
          0
        ) / locales.length
      );
      console.timeEnd('time');
    }
  });
} else {
  initProcessors().then(() => {
    process.send({ type: 'gettask' });
    process.on('message', async (msg) => {
      if (msg.type === 'exit') {
        process.exit(0);
      } else if (msg.type === 'trainandvalidate') {
        await trainAndValidate(msg.locale);
        process.send({ type: 'gettask' });
      }
    });
  });
}
