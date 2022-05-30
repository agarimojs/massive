const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { Bench } = require('@agarimo/bench');
const { initProcessors, getProcessor } = require('@agarimo/languages');
const { NeuralBig: Neural } = require('@agarimo/neural');
const { loadMassive, augmentNgrams } = require('./helpers');

const DATASET_PATH = './data';
const MODELS_PATH = './models';
const allowedLocales = undefined;
const useAnnot = false;
const useCache = false;

function execFn({ net, data }) {
  let good = 0;
  for (let i = 0; i < data.length; i += 1) {
    const item = data[i];
    const classifications = net.run(item.utterance);
    if (classifications[0].intent === item.intent) {
      good += 1;
    }
  }
  return `Good ${good} of ${data.length} Accuracy: ${
    (good / data.length) * 100
  }`;
}

(async () => {
  await initProcessors();
  const locales =
    allowedLocales ||
    fs
      .readdirSync(DATASET_PATH)
      .filter((file) => file.endsWith('.jsonl'))
      .map((file) => file.replace('.jsonl', ''));
  let totalTransactionsPerSecond = 0;
  for (let i = 0; i < locales.length; i += 1) {
    const locale = locales[i];
    const model = JSON.parse(
      zlib.inflateSync(
        fs.readFileSync(
          path.join(
            MODELS_PATH,
            `${locale}_${useAnnot ? 'annot_' : ''}model.zjson`
          )
        )
      )
    );
    const processor = getProcessor(locale.slice(0, 2));
    const corpus = loadMassive(DATASET_PATH, locale);
    const data = [];
    corpus.data.forEach((item) => {
      const tests = useAnnot ? item.annotTests : item.tests;
      data.push(
        ...tests.map((test) => ({ utterance: test, intent: item.intent }))
      );
    });
    const finalProcessor = (str) => augmentNgrams(processor(str));
    const net = new Neural({ processor: finalProcessor });
    net.fromJSON(model);
    net.useCache = useCache;
    const bench = new Bench({ transactionsPerRun: data.length });
    bench.add('exec', execFn, () => ({ net, data, processor }));
    // eslint-disable-next-line no-await-in-loop
    const result = await bench.measure(bench.algorithms[0]);
    totalTransactionsPerSecond += result.transactionsPerSecond;
    console.log(locale);
    console.log(result);
  }
  console.log(
    `Total transactions per second: ${
      totalTransactionsPerSecond / locales.length
    }`
  );
})();
