import mmdeploy.RotatedDetector;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.Graphics;
import java.io.File;
import java.io.IOException;
import java.lang.Math;

/** @description: this is a class for RotatedDetection java demo. */
public class RotatedDetection {

    /** The main function for RotatedDetection Java demo.
     * @param deviceName: the device name of the demo.
     * @param modelPath: the rotated detection model path.
     * @param imagePath: the image path.
     */
    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 3) {
            System.out.println("usage:\njava RotatedDetection deviceName modelPath imagePath");
            return;
        }
        String deviceName = args[0];
        String modelPath = args[1];
        String imagePath = args[2];

        // create rotated detector
        RotatedDetector rotatedDetector = null;
        try {
            rotatedDetector = new RotatedDetector(modelPath, deviceName, 0);

            // load image
            BufferedImage srcImg = ImageIO.read(new File(imagePath));
            Mat img = Utils.bufferedImage2Mat(srcImg);

            // apply rotated detector
            RotatedDetector.Result[] result = rotatedDetector.apply(img);

            // print results
            Graphics ghandle = srcImg.createGraphics();
            for (int i = 0; i < result.length; i++) {
                RotatedDetector.Result value = result[i];
                float cx, cy, w, h, angle;
                cx = value.rbbox[0];
                cy = value.rbbox[1];
                w = value.rbbox[2];
                h = value.rbbox[3];
                angle = value.rbbox[4];
                float wx = w / 2 * (float)Math.cos(angle);
                float wy = w / 2 * (float)Math.sin(angle);
                float hx = -h / 2 * (float)Math.sin(angle);
                float hy = h / 2 * (float)Math.cos(angle);
                System.out.printf("box %d, score %.2f, point1: (%.2f, %.2f), point2: (%.2f, %.2f), point3: (%.2f, %.2f), point4: (%.2f, %.2f)\n",
                                  i, value.score, cx - wx - hx, cy - wy - hy, cx + wx - hx, cy + wy - hy, cx + wx + hx, cy + wy + hy, cx - wx + hx, cy - wy + hy);

                // skip rotated detections less than specified score threshold
                if (value.score < 0.1) {
                    continue;
                }
                ghandle.setColor(new Color(0, 255, 0));
                int[] polygonX = new int[] {(int)(cx - wx - hx), (int)(cx + wx - hx), (int)(cx + wx + hx), (int)(cx - wx + hx)};
                int[] polygonY = new int[] {(int)(cy - wy - hy), (int)(cy + wy - hy), (int)(cy + wy + hy), (int)(cy - wy + hy)};
                ghandle.drawPolygon(polygonX, polygonY, 4);
            }
            ghandle.dispose();
            ImageIO.write(srcImg, "png", new File("output_rotated_detection.png"));
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release rotated detector
            if (rotatedDetector != null) {
                rotatedDetector.release();
            }
        }
    }
}
