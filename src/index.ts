import Stats from 'stats.js';
const cv = require('opencv');
const bodyPix = require('@tensorflow-models/body-pix');
import config from './config';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const video = document.getElementById('video') as HTMLVideoElement;

const stats = new Stats();
let net;
let time = 0, errorCount = 0;

async function initCamera() {
    if (!navigator.mediaDevices) throw new Error("Media devices not available");
    const stream = await navigator.mediaDevices.getUserMedia(config.media.load);
    video.srcObject = stream;
    video.play();
    await new Promise((resolve, _) => { video.onloadeddata = resolve; });
    await new Promise((resolve, _) => { setTimeout(resolve, config.media.initialDelay); });
}

async function initModel() {
    return await bodyPix.load(config.bodypix.load);
}

function ensureLoad() {
    if (document.readyState === 'complete') return Promise.resolve();
    return new Promise((resolve, _) => {
        window.addEventListener('load', resolve);
    });
}

async function loop() {
    stats.begin();
    try {
        await tick();
    } catch (e) {
        if (e instanceof Error) {
            console.error(e);
        } else if (typeof e === 'number') {
            console.error(cv.exceptionFromPtr(e).msg);
        } else if (typeof e === 'string') {
            console.error(cv.exceptionFromPtr(Number(e.split(' ')[0])).msg);
        } else {
            console.error(e);
        }
        errorCount++;
    }
    if (++time === 2400) time = 0;
    stats.end();
    if (errorCount < config.maxErrorCount) {
        requestAnimationFrame(loop);
    } else {
        alert(`The program aborted (${errorCount} errors reported to your console)`);
    }
}

(async function () {
    [net] = await Promise.all([
        initModel(),
        initCamera(),
        ensureLoad(),
    ]);
    await initCV();
    document.getElementById('splash').classList.add('loaded');
    if (config.stats.show) document.body.appendChild(stats.dom);
    loop();
})();

let h, w;
let pose;
let sourceCanvas: HTMLCanvasElement;
let sourceRGBA, sourceRGB;
let ones;
let segmMask1, segmMask255, segmMaskNeg255;
let fgSegmRGB, fgSegmHSV, fgSegmHSVVec, fgSegmHThresholdMask1, fgSegmHThresholdMask2;
let bgRGB;

async function initCV() {
    h = Math.floor(video.videoHeight * config.resolution.height);
    w = Math.floor(video.videoWidth * config.resolution.width);
    console.log(`Initialized with size = (${w} x ${h})`)
    sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = w;
    sourceCanvas.height = h;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    sourceRGBA = new cv.Mat(h, w, cv.CV_8UC4);
    sourceRGB = new cv.Mat(h, w, cv.CV_8UC3);
    ones = new cv.Mat.ones(h, w, cv.CV_8UC1);
    segmMask1 = new cv.Mat(h, w, cv.CV_8UC1);
    segmMask255 = new cv.Mat(h, w, cv.CV_8UC1);
    segmMaskNeg255 = new cv.Mat(h, w, cv.CV_8UC1);
    fgSegmRGB = new cv.Mat(h, w, cv.CV_8UC3);
    fgSegmHSV = new cv.Mat(h, w, cv.CV_8UC3);
    fgSegmHSVVec = new cv.MatVector();
    fgSegmHThresholdMask1 = new cv.Mat(h, w, cv.CV_8UC1);
    fgSegmHThresholdMask2 = new cv.Mat(h, w, cv.CV_8UC1);
    bgRGB = new cv.Mat(h, w, cv.CV_8UC3);
}

async function tick() {
    // 1. Copy frame(Video -> Canvas -> Uint8Array -> cv.Mat[RBGA] -> cv.Mat[RGB])/
    const ctx = sourceCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, w, h);
    sourceRGBA.data.set(ctx.getImageData(0, 0, w, h).data);
    cv.cvtColor(sourceRGBA, sourceRGB, cv.COLOR_RGBA2RGB);

    // 2. Segmentation
    if (time % config.updateFrequency.bodypix === 0) {
        const segment = await net.segmentPerson(sourceCanvas, config.bodypix.estimate);
        pose = segment.allPoses[0];
        segmMask1.data.set(segment.data);
        cv.multiply(segmMask1, ones, segmMask255, 255);
        cv.bitwise_not(segmMask255, segmMaskNeg255);
    }
    sourceRGB.copyTo(fgSegmRGB, segmMask255);
    fgSegmRGB.setTo(new cv.Scalar(0, 0, 0), segmMaskNeg255);

    const d = time * config.speed.color % 180;
    cv.cvtColor(fgSegmRGB, fgSegmHSV, cv.COLOR_RGB2HSV);
    cv.split(fgSegmHSV, fgSegmHSVVec);
    const fgSegmHue = fgSegmHSVVec.get(0);
    cv.threshold(fgSegmHue, fgSegmHThresholdMask1, 179 - d, d, cv.THRESH_BINARY_INV);
    cv.threshold(fgSegmHue, fgSegmHThresholdMask2, 179 - d, 180 - d, cv.THRESH_BINARY);
    cv.add(fgSegmHue, fgSegmHThresholdMask1, fgSegmHue);
    cv.subtract(fgSegmHue, fgSegmHThresholdMask2, fgSegmHue);
    cv.merge(fgSegmHSVVec, fgSegmHSV);

    cv.cvtColor(fgSegmHSV, fgSegmRGB, cv.COLOR_HSV2RGB);
    fgSegmRGB.copyTo(sourceRGB, segmMask255);
    cv.imshow('canvas', sourceRGB);

    //cv.bitwise_and(sourceRGB, sourceRGB, fgSegmRGB, segmMask255);
    //cv.bitwise_and(sourceRGB, sourceRGB, bgSegmRGB, segmMaskNeg255);
    //bgSegmRGB.setTo(new cv.Scalar(0, 0, 0), segmMask255);
}
