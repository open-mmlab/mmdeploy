using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploy;
using System.Linq;

namespace pose_tracker
{
    internal class Program
    {
        static class CocoSkeleton
        {
            public static List<(int, int)> Skeleton = new List<(int, int)>
            {
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
                (5, 6),   (5, 7),   (6, 8),   (7, 9),   (8, 10),  (1, 2),  (0, 1),
                (0, 2),   (1, 3),   (2, 4),   (3, 5),   (4, 6)
            };

            public static List<Scalar> Palette = new List<Scalar>
            {
                new Scalar(255, 128, 0),   new Scalar(255, 153, 51),  new Scalar(255, 178, 102),
                new Scalar(230, 230, 0),   new Scalar(255, 153, 255), new Scalar(153, 204, 255),
                new Scalar(255, 102, 255), new Scalar(255, 51, 255),  new Scalar(102, 178, 255),
                new Scalar(51, 153, 255),  new Scalar(255, 153, 153), new Scalar(255, 102, 102),
                new Scalar(255, 51, 51),   new Scalar(153, 255, 153), new Scalar(102, 255, 102),
                new Scalar(51, 255, 51),   new Scalar(0, 255, 0),     new Scalar(0, 0, 255),
                new Scalar(255, 0, 0),     new Scalar(255, 255, 255),
            };

            public static List<int> LinkColor = new List<int>
            {
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            };

            public static List<int> PointColor = new List<int>
            {
                16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
            };
        }

        static bool Visualize(OpenCvSharp.Mat frame, PoseTrackerOutput result, int long_edge,
            int frame_id, bool with_bbox)
        {
            var skeleton = CocoSkeleton.Skeleton;
            var palette = CocoSkeleton.Palette;
            var link_color = CocoSkeleton.LinkColor;
            var point_color = CocoSkeleton.PointColor;
            float scale = 1;
            if (long_edge != 0)
            {
                scale = (float)long_edge / (float)Math.Max(frame.Cols, frame.Rows);
            }
            if (scale != 1)
            {
                Cv2.Resize(frame, frame, new Size(), scale, scale);
            }
            else
            {
                frame = frame.Clone();
            }

            Action<List<float>, Scalar> drawBbox = (bbox, color) =>
            {
                for (int i = 0; i < bbox.Count; i++)
                {
                    bbox[i] *= scale;
                }
                Cv2.Rectangle(frame, new OpenCvSharp.Point(bbox[0], bbox[1]),
                    new OpenCvSharp.Point(bbox[2], bbox[3]), color);
            };

            for (int i = 0; i < result.Count; i++)
            {
                PoseTrack pt = result.Results[i];
                for (int j = 0; j < pt.Keypoints.Count; j++)
                {
                    Pointf p = pt.Keypoints[j];
                    p.X *= scale;
                    p.Y *= scale;
                    pt.Keypoints[j] = p;
                }
                float score_thr = 0.5f;
                int[] used = new int[pt.Keypoints.Count * 2];
                for (int j = 0; j < skeleton.Count; j++)
                {
                    int u = skeleton[j].Item1;
                    int v = skeleton[j].Item2;
                    if (pt.Scores[u] > score_thr && pt.Scores[v] > score_thr)
                    {
                        used[u] = used[v] = 1;
                        var p_u = new OpenCvSharp.Point(pt.Keypoints[u].X, pt.Keypoints[u].Y);
                        var p_v = new OpenCvSharp.Point(pt.Keypoints[v].X, pt.Keypoints[v].Y);
                        Cv2.Line(frame, p_u, p_v, palette[link_color[j]], 1, LineTypes.AntiAlias);
                    }
                }
                for (int j = 0; j < pt.Keypoints.Count; j++)
                {
                    if (used[j] == 1)
                    {
                        var p = new OpenCvSharp.Point(pt.Keypoints[j].X, pt.Keypoints[j].Y);
                        Cv2.Circle(frame, p, 1, palette[point_color[j]], 2, LineTypes.AntiAlias);
                    }
                }
                if (with_bbox)
                {
                    var bbox = new List<float> { pt.Bbox.Left, pt.Bbox.Top, pt.Bbox.Right, pt.Bbox.Bottom };
                    drawBbox(bbox, new Scalar(0, 255, 0));
                }
            }

            Cv2.ImShow("pose_tracker", frame);
            return Cv2.WaitKey(1) != 'q';
        }
        static void CvMatToMat(OpenCvSharp.Mat cvMat, out MMDeploy.Mat mat)
        {
            mat = new MMDeploy.Mat();
            unsafe
            {
                mat.Data = cvMat.DataPointer;
                mat.Height = cvMat.Height;
                mat.Width = cvMat.Width;
                mat.Channel = cvMat.Dims;
                mat.Format = PixelFormat.BGR;
                mat.Type = DataType.Int8;
                mat.Device = null;
            }
        }

        static void PrintHelperMessage()
        {
            string message = "usage:\n pose_tracker device det_model pose_model video";
            Console.WriteLine(message);
        }

        static void Main(string[] args)
        {
            if (args.Length != 4)
            {
                PrintHelperMessage();
                Environment.Exit(1);
            }

            string device_ = args[0];
            string det_model_ = args[1];
            string pose_model_ = args[2];
            string video = args[3];

            Model det_model = new Model(det_model_);
            Model pose_model = new Model(pose_model_);
            Device device = new Device(device_);
            Context context = new Context(device);

            // initialize tracker
            PoseTracker tracker = new PoseTracker(det_model, pose_model, context);

            PoseTracker.Params param = new PoseTracker.Params();
            // set default param
            param.Init();
            // set custom param
            param.DetMinBboxSize  = 100;
            param.DetInterval = 1;
            param.PoseMaxNumBboxes = 6;
            // optionally use OKS for keypoints similarity comparison
            float[] sigmas = {0.026f, 0.025f, 0.025f, 0.035f, 0.035f, 0.079f, 0.079f, 0.072f, 0.072f,
                              0.062f, 0.062f, 0.107f, 0.107f, 0.087f, 0.087f, 0.089f, 0.089f };
            param.SetKeypointSigmas(sigmas);

            // create state
            PoseTracker.State state = tracker.CreateState(param);

            VideoCapture cap = new VideoCapture(video);
            if (!cap.IsOpened())
            {
                Console.WriteLine("failed to open video: " + video);
                Environment.Exit(1);
            }

            int frame_id = 0;
            OpenCvSharp.Mat frame = new OpenCvSharp.Mat();
            while (true)
            {
                cap.Read(frame);
                if (frame.Empty())
                {
                    break;
                }
                CvMatToMat(frame, out var mat);
                // process
                PoseTrackerOutput result = tracker.Apply(state, mat);

                // visualize
                if (!Visualize(frame, result, 0, frame_id++, true))
                {
                    break;
                }
            }

            param.DeleteKeypointSigmas();
            tracker.Close();
        }
    }
}
