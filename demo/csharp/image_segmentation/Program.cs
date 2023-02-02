using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploy;

namespace image_segmentation
{
    class Program
    {
        /// <summary>
        /// transform input
        /// </summary>
        static void CvMatToMat(OpenCvSharp.Mat[] cvMats, out MMDeploy.Mat[] mats)
        {
            mats = new MMDeploy.Mat[cvMats.Length];
            unsafe
            {
                for (int i = 0; i < cvMats.Length; i++)
                {
                    mats[i].Data = cvMats[i].DataPointer;
                    mats[i].Height = cvMats[i].Height;
                    mats[i].Width = cvMats[i].Width;
                    mats[i].Channel = cvMats[i].Dims;
                    mats[i].Format = PixelFormat.BGR;
                    mats[i].Type = DataType.Int8;
                    mats[i].Device = null;
                }
            }
        }

        static void CvWaitKey()
        {
            Cv2.WaitKey();
        }

        static Vec3b[] GenPalette(int classes)
        {
            Random rnd = new Random(0);
            Vec3b[] palette = new Vec3b[classes];
            for (int i = 0; i < classes; i++)
            {
                byte v1 = (byte)rnd.Next(0, 255);
                byte v2 = (byte)rnd.Next(0, 255);
                byte v3 = (byte)rnd.Next(0, 255);
                palette[i] = new Vec3b(v1, v2, v3);
            }
            return palette;
        }

        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("usage:\n  image_segmentation deviceName modelPath imagePath\n");
                Environment.Exit(1);
            }

            string deviceName = args[0];
            string modelPath = args[1];
            string imagePath = args[2];

            // 1. create handle
            Segmentor handle = new Segmentor(modelPath, deviceName, 0);

            // 2. prepare input
            OpenCvSharp.Mat[] imgs = new OpenCvSharp.Mat[1] { Cv2.ImRead(imagePath, ImreadModes.Color) };
            CvMatToMat(imgs, out var mats);

            // 3. process
            List<SegmentorOutput> output = handle.Apply(mats);

            // 4. show result
            OpenCvSharp.Mat colorMask = new OpenCvSharp.Mat(output[0].Height, output[0].Width, MatType.CV_8UC3, new Scalar());
            Vec3b[] palette = GenPalette(output[0].Classes);
            unsafe
            {
                byte* data = colorMask.DataPointer;
                fixed (int* _label = output[0].Mask)
                {
                    int* label = _label;
                    for (int i = 0; i < output[0].Height; i++)
                    {
                        for (int j = 0; j < output[0].Width; j++)
                        {
                            data[0] = palette[*label][0];
                            data[1] = palette[*label][1];
                            data[2] = palette[*label][2];
                            data += 3;
                            label++;
                        }
                    }
                }
            }
            colorMask = imgs[0] * 0.5 + colorMask * 0.5;

            Cv2.NamedWindow("mmseg", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmseg", colorMask);
            CvWaitKey();

            handle.Close();
        }
    }
}
