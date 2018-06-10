/*
 *    Copyright (C) 2017 MINDORKS NEXTGEN PRIVATE LIMITED
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.mindorks.TF_Siamese;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.mindorks.tensorflowexample.R;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    private ImageView img_1, img_2, which;
    private static final int INPUT_SIZE = 224;
    private int PICK_IMAGE_REQUEST = 1;

    private Bitmap bmp_1, bmp_2, which_bmp;
    private static final String MODEL_FILE = "file:///android_asset/siamese.pb";
    private Executor executor = Executors.newSingleThreadExecutor();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button img_picker_1, img_picker_2, compute_similarity;
        final TextView result;
        final TensorFlowInferenceInterface inferenceInterface;
        Context ctx = getApplicationContext();
        AssetManager am = ctx.getAssets();
        inferenceInterface = new TensorFlowInferenceInterface(am, MODEL_FILE);
        img_picker_1 = (Button)findViewById(R.id.img_1_picker);
        img_picker_2 = (Button)findViewById(R.id.img_2_picker);
        compute_similarity = (Button)findViewById(R.id.similarity_check);

        img_1 = (ImageView)findViewById(R.id.imageView1);
        img_2 = (ImageView)findViewById(R.id.imageView2);

        result = (TextView)findViewById(R.id.result);

        img_picker_1.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                Intent pickerPhotoIntent = new Intent();
                pickerPhotoIntent.setType("image/*");
                pickerPhotoIntent.setAction(Intent.ACTION_GET_CONTENT);
                which = img_1;
                which_bmp = bmp_1;
                startActivityForResult(Intent.createChooser(pickerPhotoIntent, "Select photo"), PICK_IMAGE_REQUEST);
            }
        });
        img_picker_2.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                Intent pickerPhotoIntent = new Intent();
                pickerPhotoIntent.setType("image/*");
                pickerPhotoIntent.setAction(Intent.ACTION_GET_CONTENT);
                which = img_2;
                which_bmp = bmp_2;
                startActivityForResult(Intent.createChooser(pickerPhotoIntent, "Select photo"), PICK_IMAGE_REQUEST);
            }
        });
        compute_similarity.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                bmp_1 = ((BitmapDrawable)img_1.getDrawable()).getBitmap();
                bmp_2 = ((BitmapDrawable)img_2.getDrawable()).getBitmap();
                int bmp_arr_1[] = new int[bmp_1.getHeight() * bmp_1.getWidth()];
                int bmp_arr_2[] = new int[bmp_2.getHeight() * bmp_2.getWidth()];
                bmp_1.getPixels(bmp_arr_1, 0, bmp_1.getWidth(), 0, 0, bmp_1.getWidth(), bmp_1.getHeight());
                bmp_2.getPixels(bmp_arr_2, 0, bmp_2.getWidth(), 0, 0, bmp_2.getWidth(), bmp_2.getHeight());
                float[] bmp_float_1 = normalize(bmp_arr_1);
                float[] bmp_float_2 = normalize(bmp_arr_2);
                String[] outputs = new String[]{"output_node0"};
                inferenceInterface.feed("input_1", bmp_float_1, 1, INPUT_SIZE, INPUT_SIZE, 3);
                inferenceInterface.feed("input_2", bmp_float_2, 1, INPUT_SIZE, INPUT_SIZE, 3);
                inferenceInterface.run(outputs);
                String res = "Similarity: 0.5";
                result.setText(res);
            }
        });
    }

    public float[] normalize(int[] arr)
    {
        float[] normalized_arr = new float[INPUT_SIZE * INPUT_SIZE];
        for (int i = 0; i < arr.length; i++)
            normalized_arr[i] = ((float)arr[i]) / 255;
        return normalized_arr;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && null != data && data.getData() != null) {
            Uri uri = data.getData();
            try {
                Bitmap bmp = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                Bitmap reshaped_bmp = reshapeBitmap((bmp));
                which.setImageBitmap(reshaped_bmp);
                which_bmp = reshaped_bmp;
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public Bitmap reshapeBitmap(Bitmap bmp) {
        int width = bmp.getWidth();
        int height = bmp.getHeight();
        float scaleWidth = ((float)INPUT_SIZE) / width;
        float scaleHeight = ((float)INPUT_SIZE) / height;
        Matrix m = new Matrix();
        m.postScale(scaleWidth, scaleHeight);
        Bitmap res = Bitmap.createBitmap(bmp, 0, 0, width, height, m, false);
        return res;
    }
}

