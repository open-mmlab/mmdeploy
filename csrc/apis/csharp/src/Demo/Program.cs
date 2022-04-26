using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using MMDeploySharp;
using OpenCvSharp;

namespace Demo
{
    class Program
    {
        /// <summary>
        /// transform input
        /// </summary>
        static void CvMatToMmMat(Mat[] cvMats, out MmMat[] mats)
        {
            mats = new MmMat[cvMats.Length];
            unsafe
            {
                for (int i = 0; i < cvMats.Length; i++)
                {
                    mats[i].Data = cvMats[i].DataPointer;
                    mats[i].Height = cvMats[i].Height;
                    mats[i].Width = cvMats[i].Width;
                    mats[i].Channel = cvMats[i].Dims;
                    mats[i].Format = MmPixelFormat.MM_BGR;
                    mats[i].Type = MmDataType.MM_INT8;
                }
            }
        }

        static void CvWaitKey()
        {
            Cv2.WaitKey();
        }

        static void TestPoseDetector()
        {
            // 1. create handle
            PoseDetector handle = new PoseDetector(@"D:\test_model\hrnet", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\hrnet\human-pose.jpg", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<PoseDetectorOutput> output = handle.Apply(mats);

            // 4. show result
            Vec3b[] gen_palette(int n)
            {
                Random rnd = new Random(2);
                Vec3b[] _palette = new Vec3b[n];
                for (int i = 0; i < n; i++)
                {
                    byte v1 = (byte)rnd.Next(0, 255);
                    byte v2 = (byte)rnd.Next(0, 255);
                    byte v3 = (byte)rnd.Next(0, 255);
                    _palette[i] = new Vec3b(v1, v2, v3);
                }
                return _palette;
            }
            Vec3b[] palette = gen_palette(output[0].Count);
            int index = 0;
            foreach (var box in output[0].Results)
            {
                for (int i = 0; i < box.Points.Count; i++)
                {
                    Cv2.Circle(imgs[0], (int)box.Points[i].X, (int)box.Points[i].Y, 1, new Scalar(palette[index][0], palette[index][1], palette[index][2]), 2);
                }
                index++;
            }
            Cv2.NamedWindow("pose", WindowFlags.GuiExpanded);
            Cv2.ImShow("pose", imgs[0]);
            CvWaitKey();

            handle.Close();
        }

        static void TestClassifier()
        {

            // 1. create handle
            Classifier handle = new Classifier(@"D:\test_model\mmcls", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\mmcls\ILSVRC2012_val_00000174.JPEG", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<ClassifierOutput> output = handle.Apply(mats);

            // 4. show result
            int idx = 1;
            int offset = 50;
            foreach (var obj in output[0].Results)
            {
                string text = string.Format("Top-{0}-label: {1}, score: {2}", idx, obj.LabelId, obj.Score);
                Cv2.PutText(imgs[0], text, new Point(5, offset), HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 2);
                idx++;
                offset += 30;
            }
            Cv2.NamedWindow("mmcls", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmcls", imgs[0]);
            CvWaitKey();

            handle.Close();
        }

        static void TestDetector()
        {
            // 1. create handle
            Detector handle = new Detector(@"D:\test_model\mask_rcnn", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\mask_rcnn\demo.jpg", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<DetectorOutput> output = handle.Apply(mats);

            // 4. show result
            foreach (var obj in output[0].Results)
            {
                if (obj.Score > 0.3)
                {
                    if (obj.HasMask)
                    {
                        Mat imgMask = new Mat(obj.Mask.Height, obj.Mask.Width, MatType.CV_8UC1, obj.Mask.Data);
                        float x0 = Math.Max((float)Math.Floor(obj.BBox.Left) - 1, 0f);
                        float y0 = Math.Max((float)Math.Floor(obj.BBox.Top) - 1, 0f);
                        Rect roi = new Rect((int)x0, (int)y0, obj.Mask.Width, obj.Mask.Height);
                        Cv2.Split(imgs[0], out Mat[] ch);
                        int col = 0;
                        Cv2.BitwiseOr(imgMask, ch[col][roi], ch[col][roi]);
                        Cv2.Merge(ch, imgs[0]);
                    }

                    Cv2.Rectangle(imgs[0], new Point((int)obj.BBox.Left, (int)obj.BBox.Top), new Point((int)obj.BBox.Right, obj.BBox.Bottom), new Scalar(0, 255, 0));
                }
            }
            Cv2.NamedWindow("mmdet", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmdet", imgs[0]);
            CvWaitKey();

            handle.Close();
        }

        static void TestRestorer()
        {
            // 1. create handle
            Restorer handle = new Restorer(@"D:\test_model\srcnn", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\srcnn\baby.png", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<RestorerOutput> output = handle.Apply(mats);

            // 4. show result
            Mat sr_img = new Mat(output[0].Height, output[0].Width, MatType.CV_8UC3, output[0].Data);
            Cv2.CvtColor(sr_img, sr_img, ColorConversionCodes.RGB2BGR);
            Cv2.NamedWindow("sr", WindowFlags.GuiExpanded);
            Cv2.ImShow("sr", sr_img);
            CvWaitKey();

            handle.Close();
        }

        static void TestSegmentor()
        {
            // 1. create handle
            Segmentor handle = new Segmentor(@"D:\test_model\fcn", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\fcn\berlin_000000_000019_leftImg8bit.png", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<SegmentorOutput> output = handle.Apply(mats);

            // 4. show result
            Vec3b[] gen_palette(int classes)
            {
                Random rnd = new Random(0);
                Vec3b[] _palette = new Vec3b[classes];
                for (int i = 0; i < classes; i++)
                {
                    byte v1 = (byte)rnd.Next(0, 255);
                    byte v2 = (byte)rnd.Next(0, 255);
                    byte v3 = (byte)rnd.Next(0, 255);
                    _palette[i] = new Vec3b(v1, v2, v3);
                }
                return _palette;
            }

            Mat color_mask = new Mat(output[0].Height, output[0].Width, MatType.CV_8UC3, new Scalar());
            Vec3b[] palette = gen_palette(output[0].Classes);
            unsafe
            {
                byte* data = color_mask.DataPointer;
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
            color_mask = imgs[0] * 0.5 + color_mask * 0.5;

            Cv2.NamedWindow("mmseg", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmseg", color_mask);
            CvWaitKey();

            handle.Close();
        }

        static void TestTextDetector()
        {
            // 1. create handle
            MMDeploySharp.TextDetector handle = new MMDeploySharp.TextDetector(@"D:\test_model\dbnet", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\dbnet\demo_text_det.jpg", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<TextDetectorOutput> output = handle.Apply(mats);

            // 4. show result
            foreach (var detect in output[0].Results)
            {
                for (int i = 0; i < 4; i++)
                {
                    int sp = i;
                    int ep = (i + 1) % 4;
                    Cv2.Line(imgs[0], new Point((int)detect.BBox[sp].X, (int)detect.BBox[sp].Y),
                        new Point((int)detect.BBox[ep].X, (int)detect.BBox[ep].Y), new Scalar(0, 255, 0));
                }
            }

            Cv2.NamedWindow("ocr-det", WindowFlags.GuiExpanded);
            Cv2.ImShow("ocr-det", imgs[0]);
            CvWaitKey();

            handle.Close();
        }

        static void TestTextRecognizer()
        {
            // 1. create handle
            TextRecognizer handle = new TextRecognizer(@"D:\test_model\crnn", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\crnn\demo_text_recog.jpg", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

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

        static void Main()
        {
            TestClassifier();
            TestPoseDetector();
            TestDetector();
            TestSegmentor();
            TestRestorer();
            TestTextDetector();
            TestTextRecognizer();
        }
    }
}
