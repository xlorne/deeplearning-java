package com.codingapi.deeplearning.demo09.learn.controller;

import com.codingapi.deeplearning.demo09.learn.core.NeuralNetwork;
import com.codingapi.deeplearning.demo09.learn.core.NeuralNetworkBuilder;
import com.codingapi.deeplearning.demo09.learn.utils.ImageINDArray;
import lombok.extern.slf4j.Slf4j;
import net.coobird.thumbnailator.Thumbnails;
import org.apache.commons.io.FileUtils;
import org.apache.commons.net.util.Base64;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * @author lorne
 * @date 2019/12/31
 * @description
 */
@RestController
@Slf4j
public class PredictController {

    private NeuralNetwork neuralNetwork;

    public PredictController() throws IOException{
        neuralNetwork =  NeuralNetworkBuilder.load("model.bin");
    }


    @PostMapping("predict")
    public int predict(@RequestParam("data") String data) throws IOException {
//        log.info("data:{}",data);
        //图片压缩
        byte[] bytes = Base64.decodeBase64(data.split(",")[1]);
        File file = new File("image.png");
        FileUtils.writeByteArrayToFile(file,bytes);
        Thumbnails.of(file).size(28, 28).outputFormat("png").toFile(file);
        //类型转换
        BufferedImage bufferedImage =  ImageIO.read(file);
        INDArray array =  ImageINDArray.parser(bufferedImage);
        //预测
        int number =  neuralNetwork.perdictIndex(array);
        log.info("number:"+number);
        file.deleteOnExit();
        return number;
    }




}
