package com.codingapi.deeplearning.demo09.learn.utils;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * @author lorne
 * @date 2020/1/1
 * @description
 */
@Slf4j
public class ImageINDArray {


    //处理图片 + 特此缩放
    public  static INDArray parser(BufferedImage bufferedImage)  {
        BufferedImage bi = new BufferedImage(
                bufferedImage.getWidth(), bufferedImage.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY);

        bi.createGraphics().drawImage(bufferedImage, 0, 0,
                Color.WHITE, null);
        int width = bi.getWidth();
        int height = bi.getHeight();
        float[][] rgbs = new float[width][height];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int pixel = bi.getRGB(j, i);
                // 下面三行代码将一个数字转换为RGB数字
                int[] rgb = new int[3];
                rgb[0] = (pixel & 0xff0000) >> 16;
                rgb[1] = (pixel & 0xff00) >> 8;
                rgb[2] = (pixel & 0xff);

                rgbs[i][j] = (255-rgb[1])/255f;
//                System.out.println("（" + i + ","+j+")="  + rgb[0] + ","  + rgb[1] + "," + rgb[2] + "| rgbs:"+rgbs[i*27+j]);
//                System.out.print(rgbs[i][j]+"\t");
            }
//            System.out.println();
        }

        INDArray res = Nd4j.create(rgbs);
        log.info("res.shape:{}",res.shape());
        System.out.println(res);
        return res.reshape(1,28*28);
    }
}
