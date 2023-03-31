using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MMDeploy;
using OpenCvSharp.Extensions;
using ImreadModes = OpenCvSharp.ImreadModes;
using Cv2 = OpenCvSharp.Cv2;
using CvMat = OpenCvSharp.Mat;

#pragma warning disable IDE1006
#pragma warning disable IDE0044

namespace image_classification
{
    public partial class FormDemo : Form
    {
        Classifier classifier;
        string modelPath = "";
        string device = "cuda";
        int deviceId = 0;
        string imgPath = "";

        #region auto resize
        private float x;
        private float y;
        private void setTag(Control cons)
        {
            foreach (Control con in cons.Controls)
            {
                con.Tag = con.Width + ";" + con.Height + ";" + con.Left + ";" + con.Top + ";" + con.Font.Size;
                if (con.Controls.Count > 0)
                {
                    setTag(con);
                }
            }
        }
        private void setControls(float newx, float newy, Control cons)
        {
            foreach (Control con in cons.Controls)
            {
                if (con.Tag != null)
                {
                    string[] mytag = con.Tag.ToString().Split(new char[] { ';' });
                    con.Width = Convert.ToInt32(System.Convert.ToSingle(mytag[0]) * newx);
                    con.Height = Convert.ToInt32(System.Convert.ToSingle(mytag[1]) * newy);
                    con.Left = Convert.ToInt32(System.Convert.ToSingle(mytag[2]) * newx);
                    con.Top = Convert.ToInt32(System.Convert.ToSingle(mytag[3]) * newy);
                    Single currentSize = System.Convert.ToSingle(mytag[4]) * newy;
                    con.Font = new Font(con.Font.Name, currentSize, con.Font.Style, con.Font.Unit);
                    if (con.Controls.Count > 0)
                    {
                        setControls(newx, newy, con);
                    }
                }
            }
        }
        private void FormDemo_Resize(object sender, EventArgs e)
        {
            float newx = (this.Width) / x;
            float newy = (this.Height) / y;
            setControls(newx, newy, this);
        }

        #endregion

        static void CvMatToMat(CvMat[] cvMats, out Mat[] mats)
        {
            mats = new Mat[cvMats.Length];
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

        public FormDemo()
        {
            InitializeComponent();
            x = this.Width;
            y = this.Height;
            setTag(this);
        }

        private void radioButtonDeviceCpu_CheckedChanged(object sender, EventArgs e)
        {
            device = "cpu";
        }

        private void radioButtonDeviceCuda_CheckedChanged(object sender, EventArgs e)
        {
            device = "cuda";
        }

        private void buttonSelectModelPath_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog dilog = new FolderBrowserDialog();
            if (dilog.ShowDialog() == DialogResult.OK)
            {
                textBoxModelPath.Text = dilog.SelectedPath;
            }
        }

        private void buttonInitModel_Click(object sender, EventArgs e)
        {
            if (classifier != null)
            {
                classifier.Close();

            }
            classifier = null;
            textBoxStatus.Text = "init model ...";
            try
            {
                modelPath = textBoxModelPath.Text;
                classifier = new Classifier(modelPath, device, deviceId);
                textBoxStatus.ForeColor = Color.Green;
                textBoxStatus.Text = "init model success.";
            } catch
            {
                textBoxStatus.ForeColor = Color.Red;
                textBoxStatus.Text = "init model failed.";
            }
        }

        private void buttonLoadImage_Click(object sender, EventArgs e)
        {
            OpenFileDialog dilog = new OpenFileDialog
            {
                Filter = "(*.jpg;*.bmp;*.png;*.JPEG)|*.jpg;*.bmp;*.png;*.JPEG"
            };
            if (dilog.ShowDialog() == DialogResult.OK)
            {
                imgPath = dilog.FileName;
                CvMat img = Cv2.ImRead(dilog.FileName);
                Bitmap bitmap = BitmapConverter.ToBitmap(img);
                pictureBoxPicture.Image = bitmap;
            }
        }

        private void buttonInference_Click(object sender, EventArgs e)
        {
            textBoxResult.Clear();
            if (classifier == null)
            {
                MessageBox.Show("init model first");
                return;
            }

            CvMat[] imgs = new CvMat[1] { Cv2.ImRead(imgPath, ImreadModes.Color) };
            CvMatToMat(imgs, out var mats);

            try
            {
                List<ClassifierOutput> output = classifier.Apply(mats);
                int idx = 1;
                foreach (var obj in output[0].Results)
                {
                    if (obj.Score < 1e-7)
                    {
                        break;
                    }
                    string res = string.Format("Top-{0}-label: {1}, score: {2:f3}", idx, obj.Id, obj.Score);
                    if (idx == 1)
                    {
                        textBoxResult.Text = res;
                    }
                    else
                    {
                        textBoxResult.AppendText("\r\n" + res);
                    }
                    idx++;
                }
            } catch
            {
                MessageBox.Show("inference error");
            }
        }

        private void FormDemo_Load(object sender, EventArgs e)
        {
            textBoxUsage.Text = "1) select model dir" +
                "\r\n" + "2) choose device" +
                "\r\n" + "3) init model" +
                "\r\n" + "4) select image" +
                "\r\n" + "5) do inference";

            textBoxStatus.ForeColor = Color.Gray;
            textBoxStatus.Text = "model not init";
        }
    }
}
