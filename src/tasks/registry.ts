import type { EmbeddingGallery } from '../biometric/gallery'
import type { VisionTask } from './types'
import { CalibrationTask } from './calibration'
import { ClassifyTask } from './classify'
import { FaceDetectTask } from './faceDetect'
import { FaceLandmarkTask } from './faceLandmark'
import { FaceRecognizeTask } from './faceEmbed'
import { HandLandmarkTask } from './handLandmark'
import { ObjectDetectTask } from './objectDetect'
import { PalmRecognizeTask } from './palmEmbed'
import { PoseTask } from './pose'

/** 任务需要注入的外部依赖。两个识别任务各要一个自己的特征库。 */
export interface TaskDeps {
  faceGallery: EmbeddingGallery
  palmGallery: EmbeddingGallery
}

/**
 * 全部可选任务。数组顺序决定侧栏里的展示顺序。
 * 每接入一个新算法，在这里注册即可，UI 与调度器都会自动带上它。
 */
export function createTasks(deps: TaskDeps): VisionTask[] {
  return [
    new FaceDetectTask(),
    new FaceLandmarkTask(),
    new PoseTask(),
    new HandLandmarkTask(),
    new ObjectDetectTask(),
    new ClassifyTask(),
    new FaceRecognizeTask(deps.faceGallery),
    new PalmRecognizeTask(deps.palmGallery),
    new CalibrationTask(),
  ]
}
