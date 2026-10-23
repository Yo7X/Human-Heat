import {BrowserWindow, app, screen, desktopCapturer, ipcMain} from 'electron'
import path from 'path'
import * as ort from 'onnxruntime-node'
import sharp from 'sharp'
import fs from 'fs'

let win;
function createMainWindow() {
    win = new BrowserWindow({
        height: 600,
        x: 900,
        width: 400,
        webPreferences: {
            preload: path.join(app.getAppPath(), 'preload_overlay.js'),
            devTools: true,
        }
    })

    win.loadFile(path.join(app.getAppPath(), 'renderer/index.html'))
}

let diswin
function createOverlayWindow() {
    const display = screen.getPrimaryDisplay()
    diswin = new BrowserWindow({
        height: 500,
        width: 500,
        // x: display.bounds.x,
        // y: display.bounds.y,
        // width: display.bounds.width,
        // height: display.bounds.height,
        // transparent: true,
        // frame: false,
        // alwaysOnTop: true,
        // skipTaskbar: true,
        // focusable: false,

        webPreferences: {
            preload: path.join(app.getAppPath(), 'preload_overlay.js'),
        }
    })

    //diswin.setIgnoreMouseEvents(true, {forward: true})

    diswin.loadFile(path.join(app.getAppPath(), 'renderer', 'overlay.html'))
}

async function screenCap() {
    const sources = await desktopCapturer.getSources({
        types: ['screen'],
        thumbnailSize: {
            width: 1280,
            height: 720
        }
    })
    return sources[0].thumbnail
}

async function loadModel(modelPath) {
    const model = await ort.InferenceSession.create(path.join(app.getAppPath(), modelPath))
    console.log(model.inputNames, model.inputMetadata, model.outputNames, model.outputMetadata)
    return(model)
}



ipcMain.handle('censorPath', async (e, imgpath) => {
    const bodies = await findBodies(imgpath)
    const bodyTensors = []
    for (let i = 0; i < bodies.length; i++) {
        const newBodyTensor = await convertBodyImageToTensor(path.join(app.getPath('userData'), `body${i}.png`))

        const schp = await loadModel('models/schp-lip-20-int8-static.onnx')
        const segmentedBody = await segmentBody(schp, newBodyTensor)

        await exportSegmentation(segmentedBody, i)
    }
    
    win.webContents.send("detections", bodies.length)
})

app.whenReady().then(async () => {
    createMainWindow()


    // const screenshot = await screenCap()
    // fs.writeFileSync(
    //     path.join(app.getPath('userData'), 'screenshot.png'),
    //     screenshot.toPNG()
    // )

    //win.webContents.send("detections", ['detections', 's'])
})

async function findBodies(imgPath) {
    const model = await loadModel('models/resnet50_uint8.onnx')

    const screenshot = imgPath
    const tensor = await convertImageToTensor(screenshot)

    const mask = new ort.Tensor(
        'int64',
        new BigInt64Array(64 * 64).fill(1n),
        [1, 64, 64]
    )

    const results = await model.run({
        pixel_values: tensor,
        pixel_mask: mask
    })

    const detections = await exportDetections(
        screenshot,
        results,
        path.join(app.getPath('userData'), 'detections.png')
    )

    const { width: imageWidth, height: imageHeight } = await sharp(imgPath).metadata()
    for (let i = 0; i < detections.length; i++) {
        const box = detections[i]

        const left = Math.max(0, Math.min(imageWidth, Math.floor(box.x)))
        const top = Math.max(0, Math.min(imageHeight, Math.floor(box.y)))

        const right = Math.max(0, Math.min(imageWidth, Math.ceil(box.x + box.width)))
        const bottom = Math.max(0, Math.min(imageHeight, Math.ceil(box.y + box.height)))

        const width = right - left
        const height = bottom - top

        if (width <= 0 || height <= 0) continue

        const croppedBody = await sharp(imgPath).extract({
            left,
            top,
            width,
            height
        })
        .png()
        .toFile(path.join(app.getPath('userData'), `body${i}.png`))
    }

    return(detections)
}

async function convertImageToTensor(imagePath) {
    const {data} = await sharp(imagePath)
        .resize(
            640,
            640,
            {fit: 'fill'}
        )
        .removeAlpha()
        .toColorspace('srgb')
        .raw()
        .toBuffer({
            resolveWithObject:true
        })
    
    const pixels = new Float32Array(3 * 640 * 640)

    for (let i = 0; i < 640 * 640; i++) {
        pixels[i] = data[i * 3] / 255
        pixels[i + 640 * 640] = data[i * 3 + 1] / 255
        pixels[i + 2 * 640 * 640] = data[i * 3 + 2] / 255
    }

    return new ort.Tensor('float32', pixels, [1, 3, 640, 640])
}

async function exportDetections(imagePath, results, outputPath) {
    const logits = results.logits.data
    const boxes = results.pred_boxes.data

    const { width, height } = await sharp(imagePath).metadata()

    const detections = []

    for (let i = 0; i < 100; i++) {
        const scores = logits.slice(i * 92, (i + 1) * 92)

        const maxLogit = Math.max(...scores)
        const probabilities = scores.map(s => Math.exp(s - maxLogit))
        const sum = probabilities.reduce((a, b) => a + b, 0)

        let bestClass = 0
        let bestScore = 0

        for (let c = 0; c < 91; c++) {
            const score = probabilities[c] / sum

            if (score > bestScore) {
                bestScore = score
                bestClass = c
            }
        }

        if (bestScore < 0.5 || bestClass !== 1) continue

        const [cx, cy, w, h] = boxes.slice(i * 4, i * 4 + 4)

        detections.push({
            x: (cx - w / 2) * width,
            y: (cy - h / 2) * height,
            width: w * width,
            height: h * height,
            classId: bestClass,
            confidence: bestScore
        })
    }

    const svg = `
        <svg width="${width}" height="${height}"
             xmlns="http://www.w3.org/2000/svg">
            ${detections.map(d => `
                <rect
                    x="${d.x}"
                    y="${d.y}"
                    width="${d.width}"
                    height="${d.height}"
                    fill="none"
                    stroke="red"
                    stroke-width="3"
                />
                <text
                    x="${d.x}"
                    y="${Math.max(16, d.y - 5)}"
                    fill="red"
                    font-size="18"
                    font-weight="bold"
                >${(d.confidence * 100).toFixed(1)}%</text>
            `).join('')}
        </svg>
    `

    const boxedImage = await sharp(imagePath)
        .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
        .png()
        .toBuffer()

    fs.writeFileSync(
        outputPath,
        boxedImage
    )

    console.log('Saved detections to:', outputPath)

    return detections
}

async function convertBodyImageToTensor(imgpath) {
    const size = 473

    const { data } = await sharp(imgpath)
        .resize(size, size, { fit: 'fill' })
        .removeAlpha()
        .toColorspace('srgb')
        .raw()
        .toBuffer({ resolveWithObject: true })

    const pixels = new Float32Array(3 * size * size)

    const mean = [0.485, 0.456, 0.406]
    const std = [0.229, 0.224, 0.225]

    for (let i = 0; i < size * size; i++) {
        for (let c = 0; c < 3; c++) {
            pixels[c * size * size + i] =
                (data[i * 3 + c] / 255 - mean[c]) / std[c]
        }
    }

    return new ort.Tensor(
        'float32',
        pixels,
        [1, 3, size, size]
    )
}

async function segmentBody(model, tens) {
    const tensor = tens

    const results = await model.run({
        [model.inputNames[0]]: tensor
    })

    const output = results[model.outputNames[0]]

    const logits = output.data
    const [batch, classes, height, width] = output.dims

    const pixelCount = width * height
    const mask = new Uint8Array(pixelCount)

    for (let i = 0; i < pixelCount; i++) {
        let bestClass = 0
        let bestScore = -Infinity

        for (let c = 0; c < classes; c++) {
            const score = logits[c * pixelCount + i]

            if (score > bestScore) {
                bestScore = score
                bestClass = c
            }
        }

        mask[i] = bestClass
    }

    return {
        mask,
        width,
        height
    }
}

async function exportSegmentation(segmentation, bodyNumber) {
    const { mask, width, height } = segmentation

    const colors = [
        [0, 0, 0, 0],
        [128, 0, 0, 255],
        [128, 64, 0, 255],
        [128, 0, 128, 255],
        [0, 0, 128, 255],
        [0, 128, 0, 255],
        [128, 0, 64, 255],
        [0, 128, 128, 255],
        [255, 255, 0, 255],
        [255, 128, 0, 255],
        [0, 255, 255, 255],
        [255, 0, 128, 255],
        [128, 0, 255, 255],
        [255, 200, 160, 255],
        [255, 0, 0, 255],
        [255, 0, 0, 255],
        [0, 0, 255, 255],
        [0, 0, 255, 255],
        [128, 128, 128, 255],
        [128, 128, 128, 255]
    ]

    const pixels = Buffer.alloc(width * height * 4)

    for (let i = 0; i < mask.length; i++) {
        const color = colors[mask[i]]

        const offset = i * 4

        pixels[offset] = color[0]
        pixels[offset + 1] = color[1]
        pixels[offset + 2] = color[2]
        pixels[offset + 3] = color[3]
    }

    await sharp(pixels, {
        raw: {
            width,
            height,
            channels: 4
        }
    })
    .png()
    .toFile(path.join(app.getPath('userData'), `bodyHeatMap${bodyNumber}.png`))
}