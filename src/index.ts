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
let ones, saturator;
let segmMaskFg, segmMaskBg, segmMaskTransformed;
let fgSegmRGB, fgSegmHSV, fgSegmTransformed, fgSegmHSVVec, fgSegmHThresholdMask1, fgSegmHThresholdMask2;
let outputRGB;

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
    ones = cv.Mat.ones(h, w, cv.CV_8UC1);
    saturator = new cv.Mat(h, w, cv.CV_8UC1);
    cv.multiply(ones, ones, saturator, config.saturation);
    segmMaskFg = new cv.Mat(h, w, cv.CV_8UC1);
    segmMaskBg = new cv.Mat(h, w, cv.CV_8UC1);
    segmMaskTransformed = new cv.Mat(h, w, cv.CV_8UC1);
    fgSegmRGB = new cv.Mat(h, w, cv.CV_8UC3);
    fgSegmHSV = new cv.Mat(h, w, cv.CV_8UC3);
    fgSegmTransformed = new cv.Mat(h, w, cv.CV_8UC3);
    fgSegmHSVVec = new cv.MatVector();
    fgSegmHThresholdMask1 = new cv.Mat(h, w, cv.CV_8UC1);
    fgSegmHThresholdMask2 = new cv.Mat(h, w, cv.CV_8UC1);
    outputRGB = new cv.Mat(h, w, cv.CV_8UC3);
}

function beInRange(low, x, high) {
    return Math.max(low, Math.min(x, high));
}

async function tick() {
    {   // 1. Copy frame(Video -> Canvas -> Uint8Array -> cv.Mat[RBGA] -> cv.Mat[RGB])
        const ctx = sourceCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, w, h);
        sourceRGBA.data.set(ctx.getImageData(0, 0, w, h).data);
        cv.cvtColor(sourceRGBA, sourceRGB, cv.COLOR_RGBA2RGB);
    }

    {// 2. Segmentation
        if (time % config.updateFrequency.bodypix === 0) {
            const segment = await net.segmentPerson(sourceCanvas, config.bodypix.estimate);
            pose = segment.allPoses[0];
            segmMaskFg.data.set(segment.data);
            cv.multiply(segmMaskFg, ones, segmMaskFg, 255);
            cv.bitwise_not(segmMaskFg, segmMaskBg);
        }
        if (!pose || pose.keypoints[1].score < config.bodypix.threshold.eyes || pose.keypoints[2].score < config.bodypix.threshold.eyes) {
            cv.imshow('canvas', sourceRGB); // cannot detect eyes
            return;
        }
        fgSegmRGB.setTo(new cv.Scalar(0, 0, 0), segmMaskBg);
        sourceRGB.copyTo(fgSegmRGB, segmMaskFg);
    }

    {// 3. Paint
        const d = time * config.speed.color % 180;
        cv.cvtColor(fgSegmRGB, fgSegmHSV, cv.COLOR_RGB2HSV);
        cv.split(fgSegmHSV, fgSegmHSVVec);
        const fgSegmHue = fgSegmHSVVec.get(0);
        cv.threshold(fgSegmHue, fgSegmHThresholdMask1, 179 - d, d, cv.THRESH_BINARY_INV);
        cv.threshold(fgSegmHue, fgSegmHThresholdMask2, 179 - d, 180 - d, cv.THRESH_BINARY);
        cv.add(fgSegmHue, fgSegmHThresholdMask1, fgSegmHue);
        cv.subtract(fgSegmHue, fgSegmHThresholdMask2, fgSegmHue);
        const fgSegmSaturation = fgSegmHSVVec.get(1);
        cv.add(fgSegmSaturation, saturator, fgSegmSaturation);
        cv.merge(fgSegmHSVVec, fgSegmHSV);
        cv.cvtColor(fgSegmHSV, fgSegmRGB, cv.COLOR_HSV2RGB);
    }

    {// 4. Paint BG
        sourceRGB.copyTo(outputRGB);
    }

    {// 5. Make distortion
        const eyes = [pose.keypoints[1].position, pose.keypoints[2].position];
        const center = { x: (eyes[0].x + eyes[1].x) / 2, y: (eyes[0].y + eyes[1].y) / 2 };
        const nose = pose.keypoints[0].position;
        const r = Math.hypot(nose.x - center.x, nose.y - center.y);
        const th = Math.min(nose.y + r * config.faceRange, h - 1), bh = h - th; // top height, bottom height
        const delta = {
            x: beInRange(-w, config.rotationRadius.x * r * Math.cos(time * config.speed.rotation), w),
            y: beInRange(th - h, config.rotationRadius.y * r * Math.sin(time * config.speed.rotation), h - th),
        };
        // bottom
        const bottomSourceRect = new cv.Rect(0, th, w, bh);
        const bottomDestRect = new cv.Rect(0, th + delta.y, w, bh - delta.y);
        const bottomSourceROI = fgSegmRGB.roi(bottomSourceRect);
        const bottomDestROI = fgSegmTransformed.roi(bottomDestRect);
        const bottomSourceMaskROI = segmMaskFg.roi(bottomSourceRect);
        const bottomDestMaskROI = segmMaskTransformed.roi(bottomDestRect);
        const affineMat = cv.matFromArray(2, 3, cv.CV_64FC1, [1, -delta.x / bh, delta.x, 0, (bh - delta.y) / bh, 0]);
        cv.warpAffine(bottomSourceROI, bottomDestROI, affineMat, new cv.Size(w, bh - delta.y), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
        cv.warpAffine(bottomSourceMaskROI, bottomDestMaskROI, affineMat, new cv.Size(w, bh - delta.y), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
        affineMat.delete();
        bottomSourceROI.delete();
        bottomDestROI.delete();
        bottomSourceMaskROI.delete();
        bottomDestMaskROI.delete();
        // top
        const topROISize = { x: w - Math.abs(delta.x), y: th - Math.max(0, -delta.y) };
        const topSourceRect = new cv.Rect(Math.max(0, -delta.x), Math.max(0, -delta.y), topROISize.x, topROISize.y);
        const topDestRect = new cv.Rect(Math.max(0, delta.x), Math.max(0, delta.y), topROISize.x, topROISize.y);
        const topSourceROI = fgSegmRGB.roi(topSourceRect);
        const topDestROI = fgSegmTransformed.roi(topDestRect);
        const topSourceMaskROI = segmMaskFg.roi(topSourceRect);
        const topDestMaskROI = segmMaskTransformed.roi(topDestRect)
        topSourceROI.copyTo(topDestROI, topSourceMaskROI);
        topSourceMaskROI.copyTo(topDestMaskROI);
        topSourceROI.delete();
        topDestROI.delete();
        topSourceMaskROI.delete();
        topDestMaskROI.delete();
    }

    {// 6. Show
        fgSegmTransformed.copyTo(outputRGB, segmMaskTransformed)
        cv.imshow('canvas', outputRGB);
    }
}

