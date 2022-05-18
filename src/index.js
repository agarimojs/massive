const { Bench } = require('@agarimo/bench');
const fs = require('fs');
const { NeuralBig: Neural } = require('@agarimo/neural');
const { initProcessors, getProcessor } = require('@agarimo/languages');
const {
  loadMassive,
  applyInCorpus,
  replaceAcronyms,
  augmentNgrams,
  cleanCorpus,
} = require('./helpers');
const neuralSettings = require('./neural-settings.json');

const DATASET_PATH = './data';

const allLocales = fs
  .readdirSync(DATASET_PATH)
  .filter((file) => file.endsWith('.jsonl'))
  .map((file) => file.replace('.jsonl', ''));

// *********** These are the settings that you can change ********************
// Note: Well, you can also play with hyperparameters located at neural-settings.json
// The list of locales to be executed. If undefined, all locales will be used.
const allowedLocales = ['en-US'];
// When None intent is returned, which intent should it be mapped to
const noneIntent = 'general_quirky';
// If true then annot_utt is used, otherwise the normal utt will be used
const useAnnot = false;
// **************************************************************************

const locales = allowedLocales || allLocales;

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
  const locale = locales[0];
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
  applyInCorpus(corpus, replaceAcronyms, [
    'utterances',
    'tests',
    'annotUtterances',
    'annotTests',
  ]);
  const processor = getProcessor(locale);
  applyInCorpus(corpus, (str) => processor(str).join(' '), [
    'utterances',
    'tests',
  ]);
  corpus = cleanCorpus(corpus);
  applyInCorpus(corpus, (str) => augmentNgrams(str), ['utterances', 'tests']);
  const neural = new Neural(neuralSettings);
  console.time('train');
  neural.train(corpus);
  console.timeEnd('train');
  neural.useCache = false;
  const data = [];
  corpus.data.forEach((item) => {
    item.tests.forEach((test) => {
      data.push({ utterance: test, intent: item.intent });
    });
  });
  const bench = new Bench({ transactionsPerRun: data.length });
  bench.add('exec', execFn, () => ({ net: neural, data }));
  const result = await bench.measure(bench.algorithms[0]);
  console.log(result);
})();
