import mmdeploy.PixelFormat;
import mmdeploy.PointF;
import mmdeploy.DataType;
import mmdeploy.Mat;
import mmdeploy.Model;
import mmdeploy.Device;
import mmdeploy.Context;
import mmdeploy.Profiler;
import mmdeploy.PoseTracker.*;

import org.opencv.videoio.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.imgcodecs.*;
import org.opencv.highgui.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.lang.Math;

/** @description: this is a class for PoseTracker java demo. */
public class PoseTracker {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /** This function visualize the PoseTracker results.
     * @param frame: a video frame
     * @param results: results of PoseTracker
     * @param size: image size
     * @param frameID: the frame index in the video
     * @param withBbox: draw the person bbox or not
     * @return: whether the quit keyboard input received
     */
    public static boolean Visualize(org.opencv.core.Mat frame, mmdeploy.PoseTracker.Result[] results, int size,
        int frameID, boolean withBbox) {
        int skeleton[][] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1},
                {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};
        Scalar palette[] = {new Scalar(255, 128, 0), new Scalar(255, 153, 51), new Scalar(255, 178, 102),
                            new Scalar(230, 230, 0), new Scalar(255, 153, 255), new Scalar(153, 204, 255),
                            new Scalar(255, 102, 255), new Scalar(255, 51, 255), new Scalar(102, 178, 255),
                            new Scalar(51, 153, 255), new Scalar(255, 153, 153), new Scalar(255, 102, 102),
                            new Scalar(255, 51, 51), new Scalar(153, 255, 153), new Scalar(102, 255, 102),
                            new Scalar(51, 255, 51), new Scalar(0, 255, 0), new Scalar(0, 0, 255),
                            new Scalar(255, 0, 0), new Scalar(255, 255, 255)};
        int linkColor[] = {
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            };
        int pointColor[] = {16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0};

        float scale = (float)size / (float)Math.max(frame.cols(), frame.rows());
        if (scale != 1) {
            Imgproc.resize(frame, frame, new Size(), scale, scale);
        }
        else {
            frame = frame.clone();
        }
        for (int i = 0; i < results.length; i++) {
            mmdeploy.PoseTracker.Result pt = results[i];
            for (int j = 0; j < pt.keypoints.length; j++) {
                PointF p = pt.keypoints[j];
                p.x *= scale;
                p.y *= scale;
                pt.keypoints[j] = p;
            }
            float scoreThr = 0.5f;
            int used[] = new int[pt.keypoints.length * 2];
            for (int j = 0; j < skeleton.length; j++) {
                int u = skeleton[j][0];
                int v = skeleton[j][1];
                if (pt.scores[u] > scoreThr && pt.scores[v] > scoreThr) {
                    used[u] = used[v] = 1;
                    Point pointU = new Point(pt.keypoints[u].x, pt.keypoints[u].y);
                    Point pointV = new Point(pt.keypoints[v].x, pt.keypoints[v].y);
                    Imgproc.line(frame, pointU, pointV, palette[linkColor[j]], 1);
                }
            }
            for (int j = 0; j < pt.keypoints.length; j++) {
                if (used[j] == 1) {
                    Point p = new Point(pt.keypoints[j].x, pt.keypoints[j].y);
                    Imgproc.circle(frame, p, 1, palette[pointColor[j]], 2);
                }
            }
            if (withBbox) {
                float bbox[] = {pt.bbox.left, pt.bbox.top, pt.bbox.right, pt.bbox.bottom};
                for (int j = 0; j < 4; j++) {
                    bbox[j] *= scale;
                }
                Imgproc.rectangle(frame, new Point(bbox[0], bbox[1]),
                    new Point(bbox[2], bbox[3]), new Scalar(0, 255, 0));
                }
            }

            HighGui.imshow("Pose Tracker", frame);
            // Press any key to quit.
            return HighGui.waitKey(5) == -1;
    }

    /** The main function for PoseTracker Java demo.
     * @param deviceName: the device name of PoseTracker
     * @param detModelPath: the person detection model path
     * @param poseModelPath: the pose estimation model path
     * @param videoPath: the video path
     */
    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 4) {
            System.out.println("usage:\n-Dcommand needs deviceName detModel poseModel videoPath");
            return;
        }
        String deviceName = args[0];
        String detModelPath = args[1];
        String poseModelPath = args[2];
        String videoPath = args[3];

        // create pose tracker
        mmdeploy.PoseTracker poseTracker = null;
        Model detModel = new Model(detModelPath);
        Model poseModel = new Model(poseModelPath);
        Device device = new Device(deviceName, 0);
        if (detModel.handle() == -1 || poseModel.handle() == -1 || device.handle() == -1) {
            System.out.println("failed to create model or device");
            System.exit(1);
        }
        Context context = new Context();
        context.add(device);
        try {
            poseTracker = new mmdeploy.PoseTracker(detModel, poseModel, context);

            mmdeploy.PoseTracker.Params params = poseTracker.initParams();
            params.detInterval = 5;
            params.poseMaxNumBboxes = 6;
            long stateHandle = poseTracker.createState(params);
            VideoCapture cap = new VideoCapture(videoPath);
            if (!cap.isOpened()) {
                System.out.printf("failed to open video: %s", videoPath);
                System.exit(1);
            }
            int frameID = 0;
            org.opencv.core.Mat frame = new org.opencv.core.Mat();
            while (true) {
                cap.read(frame);
                System.out.printf("processing frame %d\n", frameID);
                if (frame.empty()) {
                    break;
                }
                Mat mat = Utils.cvMatToMat(frame);
                // process
                mmdeploy.PoseTracker.Result[] result = poseTracker.apply(stateHandle, mat, -1);

                // visualize
                if (!Visualize(frame, result, 1280, frameID++, true)) {
                    break;
                }
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release pose tracker
            if (poseTracker != null) {
                poseTracker.release();
            }
            System.exit(0);
        }
    }
}
