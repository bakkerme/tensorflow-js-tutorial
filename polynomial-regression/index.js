const x_vals = [];
const y_vals = [];

let a, b, c, d;

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

const loss = (pred, label) => pred.sub(label).square().mean();

function setup() {
  createCanvas(400, 400);
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^3 + bx^2 + cx + d
  const ys = xs.pow(tf.scalar(3)).mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d);
  return ys;
}

function mousePressed() {
  const x = map(mouseX, 0, width, -1, 1);
  const y = map(mouseY, 0, height, 1, -1);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  background(0);

  stroke(255);
  strokeWeight(2);
  for (var i = 0; i < x_vals.length; i++) {
    const px = map(x_vals[i], -1, 1, 0, width);
    const py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }

  tf.tidy(() => {
    if(x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }

    const curveX = [];
    for (let x = -1; x <= 1.01; x += 0.05){
      curveX.push(x);
    }

    const ys = predict(curveX);
    let curveY = ys.dataSync();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
      let x = map(curveX[i], -1, 1, 0, width);
      let y = map(curveY[i], -1, 1, height, 0);
      vertex(x, y);
    }
    endShape();
  });
}

