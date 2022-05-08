
namespace image_classification
{
    partial class FormDemo
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.buttonSelectModelPath = new System.Windows.Forms.Button();
            this.buttonInitModel = new System.Windows.Forms.Button();
            this.buttonLoadImage = new System.Windows.Forms.Button();
            this.pictureBoxPicture = new System.Windows.Forms.PictureBox();
            this.radioButtonDeviceCpu = new System.Windows.Forms.RadioButton();
            this.radioButtonDeviceCuda = new System.Windows.Forms.RadioButton();
            this.buttonInference = new System.Windows.Forms.Button();
            this.textBoxModelPath = new System.Windows.Forms.TextBox();
            this.labelModelPath = new System.Windows.Forms.Label();
            this.labelDevice = new System.Windows.Forms.Label();
            this.panelModelOptions = new System.Windows.Forms.Panel();
            this.textBoxStatus = new System.Windows.Forms.TextBox();
            this.labelStatus = new System.Windows.Forms.Label();
            this.labelModelOptions = new System.Windows.Forms.Label();
            this.panelPicture = new System.Windows.Forms.Panel();
            this.labelPicture = new System.Windows.Forms.Label();
            this.textBoxResult = new System.Windows.Forms.TextBox();
            this.labelResult = new System.Windows.Forms.Label();
            this.labelUsage = new System.Windows.Forms.Label();
            this.textBoxUsage = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxPicture)).BeginInit();
            this.panelModelOptions.SuspendLayout();
            this.panelPicture.SuspendLayout();
            this.SuspendLayout();
            //
            // buttonSelectModelPath
            //
            this.buttonSelectModelPath.Location = new System.Drawing.Point(959, 76);
            this.buttonSelectModelPath.Name = "buttonSelectModelPath";
            this.buttonSelectModelPath.Size = new System.Drawing.Size(143, 41);
            this.buttonSelectModelPath.TabIndex = 1;
            this.buttonSelectModelPath.Text = "select";
            this.buttonSelectModelPath.UseVisualStyleBackColor = true;
            this.buttonSelectModelPath.Click += new System.EventHandler(this.buttonSelectModelPath_Click);
            //
            // buttonInitModel
            //
            this.buttonInitModel.ForeColor = System.Drawing.SystemColors.ControlText;
            this.buttonInitModel.Location = new System.Drawing.Point(920, 76);
            this.buttonInitModel.Name = "buttonInitModel";
            this.buttonInitModel.Size = new System.Drawing.Size(143, 38);
            this.buttonInitModel.TabIndex = 1;
            this.buttonInitModel.Text = "init model";
            this.buttonInitModel.UseVisualStyleBackColor = true;
            this.buttonInitModel.Click += new System.EventHandler(this.buttonInitModel_Click);
            //
            // buttonLoadImage
            //
            this.buttonLoadImage.Location = new System.Drawing.Point(461, 752);
            this.buttonLoadImage.Name = "buttonLoadImage";
            this.buttonLoadImage.Size = new System.Drawing.Size(185, 50);
            this.buttonLoadImage.TabIndex = 2;
            this.buttonLoadImage.Text = "load image";
            this.buttonLoadImage.UseVisualStyleBackColor = true;
            this.buttonLoadImage.Click += new System.EventHandler(this.buttonLoadImage_Click);
            //
            // pictureBoxPicture
            //
            this.pictureBoxPicture.Location = new System.Drawing.Point(4, 3);
            this.pictureBoxPicture.Name = "pictureBoxPicture";
            this.pictureBoxPicture.Size = new System.Drawing.Size(392, 456);
            this.pictureBoxPicture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBoxPicture.TabIndex = 3;
            this.pictureBoxPicture.TabStop = false;
            //
            // radioButtonDeviceCpu
            //
            this.radioButtonDeviceCpu.AutoSize = true;
            this.radioButtonDeviceCpu.Location = new System.Drawing.Point(244, 134);
            this.radioButtonDeviceCpu.Name = "radioButtonDeviceCpu";
            this.radioButtonDeviceCpu.Size = new System.Drawing.Size(87, 35);
            this.radioButtonDeviceCpu.TabIndex = 4;
            this.radioButtonDeviceCpu.Text = "cpu";
            this.radioButtonDeviceCpu.UseVisualStyleBackColor = true;
            this.radioButtonDeviceCpu.CheckedChanged += new System.EventHandler(this.radioButtonDeviceCpu_CheckedChanged);
            //
            // radioButtonDeviceCuda
            //
            this.radioButtonDeviceCuda.AutoSize = true;
            this.radioButtonDeviceCuda.Checked = true;
            this.radioButtonDeviceCuda.Location = new System.Drawing.Point(343, 134);
            this.radioButtonDeviceCuda.Name = "radioButtonDeviceCuda";
            this.radioButtonDeviceCuda.Size = new System.Drawing.Size(100, 35);
            this.radioButtonDeviceCuda.TabIndex = 5;
            this.radioButtonDeviceCuda.TabStop = true;
            this.radioButtonDeviceCuda.Text = "cuda";
            this.radioButtonDeviceCuda.UseVisualStyleBackColor = true;
            this.radioButtonDeviceCuda.CheckedChanged += new System.EventHandler(this.radioButtonDeviceCuda_CheckedChanged);
            //
            // buttonInference
            //
            this.buttonInference.Location = new System.Drawing.Point(849, 752);
            this.buttonInference.Name = "buttonInference";
            this.buttonInference.Size = new System.Drawing.Size(185, 50);
            this.buttonInference.TabIndex = 6;
            this.buttonInference.Text = "inference";
            this.buttonInference.UseVisualStyleBackColor = true;
            this.buttonInference.Click += new System.EventHandler(this.buttonInference_Click);
            //
            // textBoxModelPath
            //
            this.textBoxModelPath.Location = new System.Drawing.Point(243, 76);
            this.textBoxModelPath.Name = "textBoxModelPath";
            this.textBoxModelPath.Size = new System.Drawing.Size(700, 38);
            this.textBoxModelPath.TabIndex = 7;
            //
            // labelModelPath
            //
            this.labelModelPath.AutoSize = true;
            this.labelModelPath.Location = new System.Drawing.Point(82, 76);
            this.labelModelPath.Name = "labelModelPath";
            this.labelModelPath.Size = new System.Drawing.Size(145, 31);
            this.labelModelPath.TabIndex = 8;
            this.labelModelPath.Text = "model path";
            this.labelModelPath.TextAlign = System.Drawing.ContentAlignment.TopRight;
            //
            // labelDevice
            //
            this.labelDevice.AutoSize = true;
            this.labelDevice.Location = new System.Drawing.Point(139, 134);
            this.labelDevice.Name = "labelDevice";
            this.labelDevice.Size = new System.Drawing.Size(88, 31);
            this.labelDevice.TabIndex = 9;
            this.labelDevice.Text = "device";
            this.labelDevice.TextAlign = System.Drawing.ContentAlignment.TopRight;
            //
            // panelModelOptions
            //
            this.panelModelOptions.BackColor = System.Drawing.SystemColors.ButtonFace;
            this.panelModelOptions.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelModelOptions.Controls.Add(this.textBoxStatus);
            this.panelModelOptions.Controls.Add(this.labelStatus);
            this.panelModelOptions.Controls.Add(this.buttonInitModel);
            this.panelModelOptions.ForeColor = System.Drawing.SystemColors.ButtonShadow;
            this.panelModelOptions.Location = new System.Drawing.Point(38, 57);
            this.panelModelOptions.Name = "panelModelOptions";
            this.panelModelOptions.Size = new System.Drawing.Size(1090, 139);
            this.panelModelOptions.TabIndex = 10;
            //
            // textBoxStatus
            //
            this.textBoxStatus.BackColor = System.Drawing.SystemColors.Control;
            this.textBoxStatus.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.textBoxStatus.ForeColor = System.Drawing.SystemColors.WindowText;
            this.textBoxStatus.Location = new System.Drawing.Point(657, 80);
            this.textBoxStatus.Name = "textBoxStatus";
            this.textBoxStatus.ReadOnly = true;
            this.textBoxStatus.Size = new System.Drawing.Size(247, 31);
            this.textBoxStatus.TabIndex = 3;
            //
            // labelStatus
            //
            this.labelStatus.AutoSize = true;
            this.labelStatus.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.labelStatus.Location = new System.Drawing.Point(556, 76);
            this.labelStatus.Name = "labelStatus";
            this.labelStatus.Size = new System.Drawing.Size(95, 31);
            this.labelStatus.TabIndex = 2;
            this.labelStatus.Text = "status: ";
            //
            // labelModelOptions
            //
            this.labelModelOptions.AutoSize = true;
            this.labelModelOptions.ForeColor = System.Drawing.SystemColors.ControlText;
            this.labelModelOptions.Location = new System.Drawing.Point(21, 35);
            this.labelModelOptions.Name = "labelModelOptions";
            this.labelModelOptions.Size = new System.Drawing.Size(185, 31);
            this.labelModelOptions.TabIndex = 11;
            this.labelModelOptions.Text = "Model Options";
            //
            // panelPicture
            //
            this.panelPicture.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelPicture.Controls.Add(this.pictureBoxPicture);
            this.panelPicture.Location = new System.Drawing.Point(354, 259);
            this.panelPicture.Name = "panelPicture";
            this.panelPicture.Size = new System.Drawing.Size(397, 464);
            this.panelPicture.TabIndex = 12;
            //
            // labelPicture
            //
            this.labelPicture.AutoSize = true;
            this.labelPicture.Location = new System.Drawing.Point(343, 239);
            this.labelPicture.Name = "labelPicture";
            this.labelPicture.Size = new System.Drawing.Size(94, 31);
            this.labelPicture.TabIndex = 13;
            this.labelPicture.Text = "Picture";
            //
            // textBoxResult
            //
            this.textBoxResult.BackColor = System.Drawing.SystemColors.Control;
            this.textBoxResult.Location = new System.Drawing.Point(770, 267);
            this.textBoxResult.Multiline = true;
            this.textBoxResult.Name = "textBoxResult";
            this.textBoxResult.ReadOnly = true;
            this.textBoxResult.Size = new System.Drawing.Size(358, 456);
            this.textBoxResult.TabIndex = 14;
            //
            // labelResult
            //
            this.labelResult.AutoSize = true;
            this.labelResult.Location = new System.Drawing.Point(757, 233);
            this.labelResult.Name = "labelResult";
            this.labelResult.Size = new System.Drawing.Size(85, 31);
            this.labelResult.TabIndex = 15;
            this.labelResult.Text = "Result";
            //
            // labelUsage
            //
            this.labelUsage.AutoSize = true;
            this.labelUsage.Location = new System.Drawing.Point(21, 233);
            this.labelUsage.Name = "labelUsage";
            this.labelUsage.Size = new System.Drawing.Size(85, 31);
            this.labelUsage.TabIndex = 16;
            this.labelUsage.Text = "Usage";
            //
            // textBoxUsage
            //
            this.textBoxUsage.Location = new System.Drawing.Point(37, 267);
            this.textBoxUsage.Multiline = true;
            this.textBoxUsage.Name = "textBoxUsage";
            this.textBoxUsage.ReadOnly = true;
            this.textBoxUsage.Size = new System.Drawing.Size(294, 454);
            this.textBoxUsage.TabIndex = 17;
            //
            // FormDemo
            //
            this.AutoScaleDimensions = new System.Drawing.SizeF(14F, 31F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1157, 829);
            this.Controls.Add(this.textBoxUsage);
            this.Controls.Add(this.labelUsage);
            this.Controls.Add(this.labelResult);
            this.Controls.Add(this.textBoxResult);
            this.Controls.Add(this.labelPicture);
            this.Controls.Add(this.panelPicture);
            this.Controls.Add(this.labelModelOptions);
            this.Controls.Add(this.labelDevice);
            this.Controls.Add(this.labelModelPath);
            this.Controls.Add(this.textBoxModelPath);
            this.Controls.Add(this.buttonInference);
            this.Controls.Add(this.radioButtonDeviceCuda);
            this.Controls.Add(this.radioButtonDeviceCpu);
            this.Controls.Add(this.buttonLoadImage);
            this.Controls.Add(this.buttonSelectModelPath);
            this.Controls.Add(this.panelModelOptions);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Name = "FormDemo";
            this.Text = "Image_classification";
            this.Load += new System.EventHandler(this.FormDemo_Load);
            this.Resize += new System.EventHandler(this.FormDemo_Resize);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxPicture)).EndInit();
            this.panelModelOptions.ResumeLayout(false);
            this.panelModelOptions.PerformLayout();
            this.panelPicture.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button buttonSelectModelPath;
        private System.Windows.Forms.Button buttonInitModel;
        private System.Windows.Forms.Button buttonLoadImage;
        private System.Windows.Forms.Button buttonInference;
        private System.Windows.Forms.PictureBox pictureBoxPicture;
        private System.Windows.Forms.RadioButton radioButtonDeviceCpu;
        private System.Windows.Forms.RadioButton radioButtonDeviceCuda;
        private System.Windows.Forms.TextBox textBoxModelPath;
        private System.Windows.Forms.Label labelModelPath;
        private System.Windows.Forms.Label labelDevice;
        private System.Windows.Forms.Panel panelModelOptions;
        private System.Windows.Forms.Label labelModelOptions;
        private System.Windows.Forms.Panel panelPicture;
        private System.Windows.Forms.Label labelPicture;
        private System.Windows.Forms.Label labelStatus;
        private System.Windows.Forms.TextBox textBoxResult;
        private System.Windows.Forms.Label labelResult;
        private System.Windows.Forms.Label labelUsage;
        private System.Windows.Forms.TextBox textBoxUsage;
        private System.Windows.Forms.TextBox textBoxStatus;
    }
}
