using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MMDeploy
{
#pragma warning disable 0649
    internal unsafe struct CPoseTrack
    {
        public Pointf* Keypoints;
        public int KeypointCount;
        public float* Scores;
        public Rect Bbox;
        public int TargetId;
    }
#pragma warning restore 0649

    /// <summary>
    /// Single tracking result of a bbox.
    /// A picture may contains multiple reuslts.
    /// </summary>
    public struct PoseTrack
    {
        /// <summary>
        /// Keypoints.
        /// </summary>
        public List<Pointf> Keypoints;

        /// <summary>
        /// Scores.
        /// </summary>
        public List<float> Scores;

        /// <summary>
        /// Bbox.
        /// </summary>
        public Rect Bbox;

        /// <summary>
        /// TargetId.
        /// </summary>
        public int TargetId;

        /// <summary>
        /// Init data.
        /// </summary>
        private void Init()
        {
            if (Keypoints == null || Scores == null)
            {
                Keypoints = new List<Pointf>();
                Scores = new List<float>();
            }
        }

        internal unsafe void Add(CPoseTrack* result)
        {
            Init();
            for (int i = 0; i < result->KeypointCount; i++)
            {
                Keypoints.Add(new Pointf(result->Keypoints[i].X, result->Keypoints[i].Y));
                Scores.Add(result->Scores[i]);
            }

            Bbox = result->Bbox;
            TargetId = result->TargetId;
        }
    }

    /// <summary>
    /// Output of PoseTracker.
    /// </summary>
    public struct PoseTrackerOutput
    {
        /// <summary>
        /// Tracking results for single image.
        /// </summary>
        public List<PoseTrack> Results;

        /// <summary>
        /// Gets number of output.
        /// </summary>
        public int Count
        {
            get { return (Results == null) ? 0 : Results.Count; }
        }

        /// <summary>
        /// Result for box level.
        /// </summary>
        /// <param name="boxRes">Box res.</param>
        public void Add(PoseTrack boxRes)
        {
            if (Results == null)
            {
                Results = new List<PoseTrack>();
            }

            Results.Add(boxRes);
        }
    }

    /// <summary>
    /// PoseTracker.
    /// </summary>
    public class PoseTracker : DisposableObject
    {
        /// <summary>
        /// Params.
        /// </summary>
        public struct Params
        {
            /// <summary>
            /// init with default value.
            /// </summary>
            public void Init()
            {
                IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(Params)));
                NativeMethods.mmdeploy_pose_tracker_default_params(ptr);
                this = Marshal.PtrToStructure<Params>(ptr);
                Marshal.DestroyStructure<Params>(ptr);
                Marshal.FreeHGlobal(ptr);
            }

            /// <summary>
            /// Sets keypoint sigmas.
            /// </summary>
            /// <param name="array">keypoint sigmas.</param>
            public void SetKeypointSigmas(float[] array)
            {
                this.KeypointSigmasSize = array.Length;
                this.KeypointSigmas = Marshal.AllocHGlobal(sizeof(float) * array.Length);
                Marshal.Copy(array, 0, this.KeypointSigmas, array.Length);
            }

            /// <summary>
            /// Release ptr.
            /// </summary>
            public void DeleteKeypointSigmas()
            {
                if (this.KeypointSigmas != null)
                {
                    Marshal.FreeHGlobal(this.KeypointSigmas);
                    this.KeypointSigmasSize = 0;
                }
            }

            /// <summary>
            /// detection interval, default = 1.
            /// </summary>
            public int DetInterval;

            /// <summary>
            /// detection label use for pose estimation, default = 0.
            /// </summary>
            public int DetLabel;

            /// <summary>
            /// detection score threshold, default = 0.5.
            /// </summary>
            public float DetThr;

            /// <summary>
            /// detection minimum bbox size (compute as sqrt(area)), default = -1.
            /// </summary>
            public float DetMinBboxSize;

            /// <summary>
            /// nms iou threshold for merging detected bboxes and bboxes from tracked targets, default = 0.7.
            /// </summary>
            public float DetNmsThr;

            /// <summary>
            /// max number of bboxes used for pose estimation per frame, default = -1.
            /// </summary>
            public int PoseMaxNumBboxes;

            /// <summary>
            /// threshold for visible key-points, default = 0.5.
            /// </summary>
            public float PoseKptThr;

            /// <summary>
            /// min number of key-points for valid poses, default = -1.
            /// </summary>
            public int PoseMinKeypoints;

            /// <summary>
            /// scale for expanding key-points to bbox, default = 1.25.
            /// </summary>
            public float PoseBboxScale;

            /// <summary>
            /// min pose bbox size, tracks with bbox size smaller than the threshold will be dropped,default = -1.
            /// </summary>
            public float PoseMinBboxSize;

            /// <summary>
            /// nms oks/iou threshold for suppressing overlapped poses, useful when multiple pose estimations
            /// collapse to the same target, default = 0.5.
            /// </summary>
            public float PoseNmsThr;

            /// <summary>
            /// keypoint sigmas for computing OKS, will use IOU if not set, default = nullptr.
            /// </summary>
            public IntPtr KeypointSigmas;

            /// <summary>
            /// size of keypoint sigma array, must be consistent with the number of key-points, default = 0.
            /// </summary>
            public int KeypointSigmasSize;

            /// <summary>
            /// iou threshold for associating missing tracks, default = 0.4.
            /// </summary>
            public float TrackIouThr;

            /// <summary>
            /// max number of missing frames before a missing tracks is removed, default = 10.
            /// </summary>
            public int TrackMaxMissing;

            /// <summary>
            /// track history size, default = 1.
            /// </summary>
            public int TrackHistorySize;

            /// <summary>
            /// weight of position for setting covariance matrices of kalman filters, default = 0.05.
            /// </summary>
            public float StdWeightPosition;

            /// <summary>
            /// weight of velocity for setting covariance matrices of kalman filters, default = 0.00625.
            /// </summary>
            public float StdWeightVelocity;

            /// <summary>
            /// params for the one-euro filter for smoothing the outputs - (beta, fc_min, fc_derivative)
            /// default = (0.007, 1, 1).
            /// </summary>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public float[] SmoothParams;
        }

        /// <summary>
        /// tracking state.
        /// </summary>
        public class State : DisposableObject
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="State"/> class.
            /// </summary>
            /// <param name="pipeline">pipeline.</param>
            /// <param name="param">param.</param>
            public State(IntPtr pipeline, Params param)
            {
                ThrowException(NativeMethods.mmdeploy_pose_tracker_create_state(pipeline, param, out _handle));
            }

            /// <inheritdoc/>
            protected override void ReleaseHandle()
            {
                NativeMethods.mmdeploy_pose_tracker_destroy_state(_handle);
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PoseTracker"/> class.
        /// </summary>
        /// <param name="detect">detect model.</param>
        /// <param name="pose">pose model.</param>
        /// <param name="context">context.</param>
        public PoseTracker(Model detect, Model pose, Context context)
        {
            ThrowException(NativeMethods.mmdeploy_pose_tracker_create(detect, pose, context, out _handle));
        }

        /// <summary>
        /// Get track information of image.
        /// </summary>
        /// <param name="state">state for video.</param>
        /// <param name="mat">input mat.</param>
        /// <param name="detect">control the use of detector.
        /// -1: use params.DetInterval, 0: don't use detector, 1: force use detector.</param>
        /// <returns>results of this frame.</returns>
        public PoseTrackerOutput Apply(State state, Mat mat, int detect = -1)
        {
            PoseTrackerOutput output = default;

            IntPtr[] states = new IntPtr[1] { state };
            Mat[] mats = new Mat[1] { mat };
            int[] detects = new int[1] { -1 };

            unsafe
            {
                CPoseTrack* results = null;
                int* resultCount = null;
                fixed (Mat* _mats = mats)
                fixed (IntPtr* _states = states)
                fixed (int* _detects = detects)
                {
                    ThrowException(NativeMethods.mmdeploy_pose_tracker_apply(_handle, _states, _mats, _detects,
                        mats.Length, &results, &resultCount));

                    FormatResult(resultCount, results, ref output, out var total);
                    ReleaseResult(results, resultCount, mats.Length);
                }
            }

            return output;
        }

        private unsafe void FormatResult(int* resultCount, CPoseTrack* results, ref PoseTrackerOutput output, out int total)
        {
            total = resultCount[0];
            for (int i = 0; i < total; i++)
            {
                PoseTrack outi = default;
                outi.Add(results);
                output.Add(outi);
                results++;
            }
        }

        private unsafe void ReleaseResult(CPoseTrack* results, int* resultCount, int count)
        {
            NativeMethods.mmdeploy_pose_tracker_release_result(results, resultCount, count);
        }

        /// <summary>
        /// Create internal state.
        /// </summary>
        /// <param name="param">instance of Params.</param>
        /// <returns>instance of State.</returns>
        public State CreateState(Params param)
        {
            State state = new State(_handle, param);
            return state;
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            // _state.Dispose();
            NativeMethods.mmdeploy_pose_tracker_destroy(_handle);
        }
    }
}
