import mmdeploy.PoseTracker;
import mmdeploy.PixelFormat;
import mmdeploy.PointF;
import mmdeploy.DataType;
import mmdeploy.Mat;
import mmdeploy.Model;
import mmdeploy.Device;
import mmdeploy.Context;
import mmdeploy.Profiler;

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

public class PoseTrack {
    public static boolean Visualize(org.opencv.core.Mat frame, PoseTracker.Result[] results, int size,
        int frame_id, boolean with_bbox){
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
        else
        {
            frame = frame.clone();
        }

        for (int i = 0; i < results.length; i++)
        {
            PoseTracker.Result pt = results[i];
            for (int j = 0; j < pt.keypoints.length; j++)
            {
                PointF p = pt.keypoints[j];
                p.x *= scale;
                p.y *= scale;
                pt.keypoints[j] = p;
            }
            float score_thr = 0.5f;
            int used[] = new int[pt.keypoints.length * 2];
            for (int j = 0; j < skeleton.length; j++)
            {
                int u = skeleton[j][0];
                int v = skeleton[j][1];
                if (pt.scores[u] > score_thr && pt.scores[v] > score_thr)
                {
                    used[u] = used[v] = 1;
                    Point p_u = new Point(pt.keypoints[u].x, pt.keypoints[u].y);
                    Point p_v = new Point(pt.keypoints[v].x, pt.keypoints[v].y);
                    Imgproc.line(frame, p_u, p_v, palette[linkColor[j]], 1);
                }
            }
            for (int j = 0; j < pt.keypoints.length; j++)
            {
                if (used[j] == 1)
                {
                    Point p = new Point(pt.keypoints[j].x, pt.keypoints[j].y);
                    Imgproc.circle(frame, p, 1, palette[pointColor[j]], 2);
                }
            }
            if (with_bbox)
            {
                float bbox[] = {pt.bbox.left, pt.bbox.top, pt.bbox.right, pt.bbox.bottom};
                for (int j = 0; j < 4; j++)
                {
                    bbox[j] *= scale;
                }
                Imgproc.rectangle(frame, new Point(bbox[0], bbox[1]),
                    new Point(bbox[2], bbox[3]), new Scalar(0, 255, 0));
                }
            }

            HighGui.imshow("Linear Blend", frame);
            return HighGui.waitKey(1) != 'q';
    }
    public static void main(String[] args) {
        // Parse arguments
        if (args.length < 4 || args.length > 5) {
            System.out.println("usage:\njava PoseTracker device_name det_model pose_model video [output]");
            return;
        }
        String deviceName = args[0];
        String detModelPath = args[1];
        String poseModelPath = args[2];
        String videoPath = args[3];
        if (args.length == 5) {
            String outputDir = args[4];
        }

        // create pose tracker
        PoseTracker poseTracker = null;
        Model detModel = new Model(detModelPath);
        Model poseModel = new Model(poseModelPath);
        Device device = new Device(deviceName, 0);
        Context context = new Context();
        context.add(0, device.device_);
        try {
            poseTracker = new PoseTracker(detModel.model_, poseModel.model_, context.context_);
            float[] smoothParam = new float[] {0.007f, 1, 1};
            float[] keypointSigmas = new float[] {0.026f, 0.025f, 0.025f, 0.035f, 0.035f, 0.079f, 0.079f, 0.072f, 0.072f,
                              0.062f, 0.062f, 0.107f, 0.107f, 0.087f, 0.087f, 0.089f, 0.089f};
            // PoseTracker.Params params = new PoseTracker.Params(1, 0, 0.5f, -1, 0.7f, -1, 0.5f, -1, 1.25f, -1, 0.5f, keypointSigmas, 17, 0.4f, 10, 1, 0.05f, 0.00625f, smoothParam, 0);
            PoseTracker.Params params = poseTracker.initParams();
            params.detMinBboxSize = 100;
            params.detInterval = 1;
            params.poseMaxNumBboxes = 6;
            params = poseTracker.setParams(params.handle, params);
            // setParamValue must contains handle.
            long paramsHandle = params.handle;
            long stateHandle = poseTracker.createState(paramsHandle);
            VideoCapture cap = new VideoCapture(videoPath);
            if (!cap.isOpened()) {
                System.out.printf("failed to open video: %s", videoPath);
            }
            int frameID = 0;
            org.opencv.core.Mat frame = new org.opencv.core.Mat();

            while (true)
            {
                cap.read(frame);
                if (frame.empty())
                {
                    break;
                }
                Mat mat = Utils.cvMatToMat(frame);
                // process
                PoseTracker.Result[] result = poseTracker.apply(stateHandle, mat, -1);

                // visualize
                if (!Visualize(frame, result, 1280, frameID++, true))
                {
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
        }
    }
}
