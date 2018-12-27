const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
function loop() {
  const values = [];
  for(let i = 0; i < 150000; i++) {
    values[i] = Math.random() * 100;
  }

  const shape = [500, 300];

  tf.tidy(() => {
    const a = tf.tensor2d(values, shape, 'int32');
    const b = tf.tensor2d(values, shape, 'int32');
    const bb = b.transpose();

    const c = a.matMul(bb);
  });

  console.log(tf.memory().numTensors);
    // a.dispose();
  // b.dispose();
  // c.dispose();
  // bb.dispose();

}



function setup() {
  // c.print();

  // const tense = tf.tensor3d(values, shape, 'int32');

  // const tense = tf.tensor([0, 0, 127, 255, 100, 50, 24, 54], [2, 2, 2], 'int32');
  // tense.print();

  // const tenseVar = tf.variable(tense);
  // console.log(tense.get(0, 1, 5));

  while(true) {
    loop();
  }
}
setup();
