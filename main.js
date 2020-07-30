import '@babel/polyfill';
import * as tf from '@tensorflow/tfjs';
import { buffer } from '@tensorflow/tfjs';

const model = tf.sequential();
const layer1 = tf.layers.dense({ units:1, inputShape:[1] });
model.add(layer1);

model.compile({
	loss: 'meanSquaredError',
	optimizer: 'sgd'
});
log('Model created.')

const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

async function start() {
	log('Training model...')
	await model.fit(xs, ys, { epochs: 250 });
	const result = model.predict(tf.tensor2d([0.1], [1, 1])).dataSync()[0]; // Input sample x
	log(result);
}
start();

function log(text) {
	document.querySelector('#result').innerText = text;
}