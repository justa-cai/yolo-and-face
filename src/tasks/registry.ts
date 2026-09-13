import type { FaceGallery } from '../face/FaceGallery'
import type { VisionTask } from './types'
import { CalibrationTask } from './calibration'
import { ClassifyTask } from './classify'
import { FaceDetectTask } from './faceDetect'
import { FaceLandmarkTask } from './faceLandmark'
import { FaceRecognizeTask } from './faceEmbed'
import { ObjectDetectTask } from './objectDetect'
import { PoseTask } from './pose'

/**
 * 全部可选任务。数组顺序决定侧栏里的展示顺序。
 * 每接入一个新算法，在这里注册即可，UI 与调度器都会自动带上它。
 *
 * 人脸识别需要一个人脸库存特征，所以由调用方注入。
 */
export function createTasks(gallery: FaceGallery): VisionTask[] {
  return [
    new FaceDetectTask(),
    new FaceLandmarkTask(),
    new PoseTask(),
    new ObjectDetectTask(),
    new ClassifyTask(),
    new FaceRecognizeTask(gallery),
    new CalibrationTask(),
  ]
}
