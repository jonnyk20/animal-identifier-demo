import React, { useState, useRef, Fragment } from "react"
import * as tf from "@tensorflow/tfjs"
import { getImageOrientation, adjustCanvas, argMax } from "../utils"
import ProgressBar from "../components/ProgressBar"
import LoadingSpinner from "../components/LoadingSpinner"
import Boxes from "../components/Boxes"
import sampleFishPhoto from "../images/rockfish.jpg"
import { ML_STATUSES, DETECTION_MODEL_URL, CLASSIFICATION_MODEL_URL  } from '../constants'
import "../styles.scss"

const classificationLabels = ["canary rockfish", "vermillion rockfish", "yelloweye rockfish"];

const RockfishDemo = () => {
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [classificationModel, setClassificationModel] = useState(null)
  const [detectionModel, setDetectionModel] = useState(null)
  const [classifiedBoxes, setClassifiedBoxes] = useState([])

  const [
    classificationDownloadProgress,
    setClassifiationDownloadProgress,
  ] = useState(0)
  const [detectionDownloadProgress, setDetectionDownloadProgress] = useState(0)
  const downloadProgress =
    0.6 * detectionDownloadProgress + 0.4 * classificationDownloadProgress

  const [hiddenSrc, setHiddenSrc] = useState(null)
  const [resizedSrc, setResizedSrc] = useState(null)
  const [error, setError] = useState(false)
  const [status, setStatus] = useState(ML_STATUSES.INITIAL)
  const [isImageReady, setIsImageReady] = useState(false)
  const [orientation, setOrientation] = useState(0)
  const [dimensions, setDimensions] = useState({})
  const inputRef = useRef()
  const hiddenRef = useRef()
  const resizedRef = useRef()
  const rotationCanvasRef = useRef()
  const hiddenCanvasRef = useRef()
  const cropRef = useRef()

  const formatScore = score => 
    (score * 100).toFixed(2)


  const classifyDetections = (boxes, classifications) => {
    const classifiedBoxes = 
      boxes.map((box, i) => {
        const classificationLabelIndex = argMax(classifications[i]);
        const score = classifications[i][classificationLabelIndex];
        const label = classificationLabels[classificationLabelIndex];

        return ({
          ...box,
          label,
          score: formatScore(score),
        })
      })
    setClassifiedBoxes(classifiedBoxes);
    setStatus(ML_STATUSES.COMPLETE);
  }

  const cropDetections = async boxes => {
    const canvases = [];
    boxes.forEach(box => {
      const { current: source } = rotationCanvasRef
      const canvas = document.createElement('canvas');
      const { x, width: w, height: h } = source.getBoundingClientRect()
      const A = box.x // x
      const B = box.y // y
      const C = w // w original
      const D = h // h original
      const E = 0
      const F = 0
      const G = w // w original (scale)
      const H = h // h original (scale)
      const ctx = canvas.getContext("2d")
      canvas.height = box.h // cropH
      canvas.width = box.w // cropW
      ctx.drawImage(source, A, B, C, D, E, F, G, H)
      canvases.push(canvas);
    })
    const classifications = await classify(canvases);
    setStatus(ML_STATUSES.CLASSIFYING);
    classifyDetections(boxes, classifications);
  }

  const drawBoxes = boxes => {
    const { current: img } = rotationCanvasRef
    const { width: imgW, height: imgH } = img
    const formattedBoxes = []
    boxes.forEach((topBox, index) => {
      const topLeft = [topBox[1] * imgW, topBox[0] * imgH]
      const bottomRight = [topBox[3] * imgW, topBox[2] * imgH]
      const boxW = bottomRight[0] - topLeft[0]
      const boxH = bottomRight[1] - topLeft[1]
      const boxX = topLeft[0]
      const boxY = topLeft[1]
      const formattedBox = {
        index,
        x: boxX,
        y: boxY,
        w: boxW,
        h: boxH,
      }
      formattedBoxes.push(formattedBox)
    })
    cropDetections(formattedBoxes)
  }

  const formatDetectionOutput = tensors => {
    const [
      raw_detection_scores,
      raw_detection_boxes,
      detection_scores,
      detection_boxes,
      num_detections,
      detection_classes,
    ] = tensors

    const boxes = []
    for (let i = 0; i < num_detections.values[0]; i++) {
      const n = i * 4
      const box = detection_boxes.values.slice(n, n + 4)
      if (detection_scores.values[i] > 0.4) {
        boxes.push(box)
      }
    }
    drawBoxes(boxes)
  }

  const warmUpModels = async (detector, classifier) => {
    setStatus(ML_STATUSES.WARMING_UP)
    try {
      const warmupResult = await detector.executeAsync(
        tf.zeros([1, 300, 300, 3])
      )
    } catch (err) {
      console.log("ERROR ON TEST RUN", err)
    }
    setStatus(ML_STATUSES.READY_FOR_DETECTION)
  }

  const loadModels = async () => {
    try {
      const detector = await tf.loadGraphModel(DETECTION_MODEL_URL, { onProgress: setDetectionDownloadProgress })
      const classifier = await tf.loadGraphModel(CLASSIFICATION_MODEL_URL, {
        onProgress: setClassifiationDownloadProgress,
      })
      setClassificationModel(classifier)
      setDetectionModel(detector)
      setModelsLoaded(true)
      warmUpModels(detector, classifier)
    } catch (err) {
      console.log("ERROR ON LOAD", err)
    }
  }

  const detect = async () => {
    const { current: img } = rotationCanvasRef
    let predictionFailed = false
    setStatus(ML_STATUSES.DETECTING)
    try {
      const tfImg = tf.browser.fromPixels(img).toFloat()
      const expanded = tfImg.expandDims(0)
      const res = await detectionModel.executeAsync(expanded)
      const detection_boxes = res[2]
      // const arr = await detection_boxes.array()
      const tensors = await Promise.all(
        res.map(async (ts, i) => {
          return await ts.buffer()
        })
      )
      formatDetectionOutput(tensors)
    } catch (err) {
      predictionFailed = true
    }
    setError(predictionFailed)
  }

  const handleLoad = () => {
    const { current: img } = hiddenRef
    const width = img.width,
      height = img.height
    const { current: canvas } = hiddenCanvasRef
    const ctx = canvas.getContext("2d")
    adjustCanvas(canvas, ctx, width, height, orientation)
    ctx.drawImage(img, 0, 0)
    setResizedSrc(canvas.toDataURL())
  }

  const classify = async canvases =>
    Promise.all(canvases.map(async canvas => {
      // convert element to tensor
      const tensorInput = tf.browser.fromPixels(canvas).toFloat()

      // resize tensor
      const reshapedInput = tf
        .image
        .resizeBilinear(tensorInput, [224, 224])
        .expandDims(0)

      // Normalize the image
      const offset = tf.scalar(127.5)
      const normalizedInput = reshapedInput.sub(offset).div(offset)
  
      // run the classifiaction
      const results = classificationModel.predict(normalizedInput)

      // get buffer from tensor to access values as TypedArray
      const resultsData = await results.buffer()
      return resultsData.values;
    }));

  const resize = () => {
    const { innerWidth: maxWidth } = window
    const { current: canvas } = rotationCanvasRef
    const ctx = canvas.getContext("2d")
    const { current: img } = resizedRef
    let { height, width } = img

    if (width > maxWidth) {
      const ratio = width / height
      width = maxWidth
      height = maxWidth / ratio
    }
    canvas.width = width
    canvas.height = height
    setDimensions({ width, height })
    ctx.drawImage(img, 0, 0, width, height)
    setStatus(ML_STATUSES.READY_FOR_DETECTION)
    setIsImageReady(true)
  }

  const handleChange = event => {
    const { files } = event.target
    if (files.length > 0) {
      const hiddenSrc = URL.createObjectURL(event.target.files[0])
      getImageOrientation(event.target.files[0], orientation => {
        setOrientation(orientation)
        setHiddenSrc(hiddenSrc)
      })
    }
  }

  const getSamplePhoto = () => {
    setResizedSrc(sampleFishPhoto)
  }

  const clearImage = () => {
    const canvas = rotationCanvasRef.current;
    const context = canvas.getContext('2d');

    context.clearRect(0, 0, canvas.width, canvas.height);
    setIsImageReady(false)
  }

  const reset = e => {
    e.stopPropagation()
    setStatus(ML_STATUSES.READY_FOR_DETECTION)
    setResizedSrc(null)
    setClassifiedBoxes([])
    clearImage()
  }

  const triggerInput = () => {
    inputRef.current.click()
  }

  const hidden = {
    display: "none",
  }

  const showProgress = downloadProgress !== 0 && downloadProgress !== 1
  const showSpinner = status === ML_STATUSES.WARMING_UP || status === ML_STATUSES.DETECTING
  const isComplete = status === ML_STATUSES.COMPLETE

  return (
    <div
      className="wrapper"
      style={isImageReady ? { ...dimensions } : {}}
    >
      <img
        id="hidden-upload-placeholder"
        src={hiddenSrc}
        ref={hiddenRef}
        style={hidden}
        onLoad={handleLoad}
      />
      <img
        id="resized-placeholder"
        src={resizedSrc}
        ref={resizedRef}
        style={hidden}
        onLoad={resize}
      />
      <canvas ref={hiddenCanvasRef} id="hidden-canvas" style={hidden} />
      <canvas
        ref={rotationCanvasRef}
        style={isImageReady ? {} : hidden}
        id="adjusted-image"
      />
      <Boxes boxes={classifiedBoxes} />
      {isImageReady && <div className="overlay" />}

      <div className="control">
          {status === ML_STATUSES.READY_FOR_DETECTION && isImageReady && (
            <button onClick={detect} className="control__button">
              Identify Rockfish
            </button>
          )}
          {showSpinner && <LoadingSpinner />}
          {status === ML_STATUSES.WARMING_UP && <div>Warming up...</div>}
          {status === ML_STATUSES.DETECTING && <div>Detecting...</div>}
          {!modelsLoaded && (
            <button onClick={loadModels} className="control__button">
              Load Model
            </button>
          )}
          {showProgress && <ProgressBar progress={downloadProgress} />}

          {!isImageReady && status === ML_STATUSES.READY_FOR_DETECTION && (
            <Fragment>
              <button
                href="#"
                onClick={triggerInput}
                className="control__button"
              >
                Upload a Photo
              </button>
              <div className="separator">- OR -</div>
              <button
                href="#"
                onClick={getSamplePhoto}
                className="control__button"
              >
                Use a Sample
              </button>
            </Fragment>
          )}
          {isComplete && error && <div>Failed to Find Fish <br /></div>}
          
          {isComplete && (
            <button onClick={reset} className="control__button">
              Reset
            </button>
          )}

          <input
            type="file"
            accept="image/*"
            capture="camera"
            onChange={handleChange}
            ref={inputRef}
            id="file-input"
            className="control__input"
          />
      </div>
      <canvas className="cropped" ref={cropRef} style={hidden} />
    </div>
  )
}

export default RockfishDemo
