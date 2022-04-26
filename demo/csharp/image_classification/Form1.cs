using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MMDeploySharp;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace image_classification
{
    public partial class Form1 : Form
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
        private void Form1_Resize(object sender, EventArgs e)
        {
            float newx = (this.Width) / x;
            float newy = (this.Height) / y;
            setControls(newx, newy, this);
        }

        #endregion

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

        public Form1()
        {
            InitializeComponent();
            x = this.Width;
            y = this.Height;
            setTag(this);
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            device = "cpu";
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            device = "cuda";
        }

        private void button1_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog dilog = new FolderBrowserDialog();
            if (dilog.ShowDialog() == DialogResult.OK)
            {
                textBox1.Text = dilog.SelectedPath;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (classifier != null)
            {
                classifier.Close();

            }
            classifier = null;
            textBox4.Text = "init model ...";
            try
            {
                modelPath = textBox1.Text;
                classifier = new Classifier(modelPath, device, deviceId);
                textBox4.ForeColor = Color.Green;
                textBox4.Text = "init model success.";
            } catch
            {
                textBox4.ForeColor = Color.Red;
                textBox4.Text = "init model failed.";
            }


        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            Console.WriteLine("device: " + device);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            OpenFileDialog dilog = new OpenFileDialog();
            dilog.Filter = "(*.jpg;*.bmp;*.png;*.JPEG)|*.jpg;*.bmp;*.png;*.JPEG";
            if (dilog.ShowDialog() == DialogResult.OK)
            {
                imgPath = dilog.FileName;
                Mat img = Cv2.ImRead(dilog.FileName);
                Bitmap bitmap = BitmapConverter.ToBitmap(img);
                pictureBox1.Image = bitmap;
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            textBox2.Clear();
            if (classifier == null)
            {
                MessageBox.Show("init model first");
                return;
            }

            Mat[] imgs = new Mat[1] { Cv2.ImRead(imgPath, ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

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
                    string res = string.Format("Top-{0}-label: {1}, score: {2:f3}", idx, obj.LabelId, obj.Score);
                    if (idx == 1)
                    {
                        textBox2.Text = res;
                    }
                    else
                    {
                        textBox2.AppendText("\r\n" + res);
                    }
                    idx++;
                }
            } catch
            {
                MessageBox.Show("inference error");
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            textBox3.Text = "1) select model dir" +
                "\r\n" + "2) choose device" +
                "\r\n" + "3) init model" +
                "\r\n" + "4) select image" +
                "\r\n" + "5) do inference";

            textBox4.ForeColor = Color.Gray;
            textBox4.Text = "model not init";
        }
    }
}
