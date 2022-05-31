using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploy;

namespace ocr_recognition
{
    class Program
    {
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
                }
            }
        }

        static void CvWaitKey()
        {
            Cv2.WaitKey();
        }

        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("usage:\n  ocr_recognition deviceName modelPath imagePath\n");
                Environment.Exit(1);
            }

            string deviceName = args[0];
            string modelPath = args[1];
            string imagePath = args[2];

            // 1. create handle
            TextRecognizer handle = new TextRecognizer(modelPath, deviceName, 0);

            // 2. prepare input
            OpenCvSharp.Mat[] imgs = new OpenCvSharp.Mat[1] { Cv2.ImRead(imagePath, ImreadModes.Color) };
            CvMatToMat(imgs, out var mats);

            // 3. process
            List<TextRecognizerOutput> output = handle.Apply(mats);

            //// 4. show result
            foreach (var box in output[0].Results)
            {
                string text = System.Text.Encoding.UTF8.GetString(box.Text);
                Cv2.PutText(imgs[0], text, new Point(20, 20), HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 1);
            }

            Cv2.NamedWindow("ocr-reg", WindowFlags.GuiExpanded);
            Cv2.ImShow("ocr-reg", imgs[0]);
            CvWaitKey();

            handle.Close();
        }
    }
}
