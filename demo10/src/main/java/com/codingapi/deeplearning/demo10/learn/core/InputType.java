package com.codingapi.deeplearning.demo10.learn.core;

import lombok.Getter;
import lombok.ToString;

/**
 * @author lorne
 * @date 2020/2/5
 * @description
 */
@Getter
@ToString
public class InputType {

    private int width;
    private int height;
    private int depth;


    private final int inputSize;


    public InputType(int inputSize){
        this.inputSize = inputSize;
    }

    public InputType(int width, int height, int depth) {
        this.width = width;
        this.height = height;
        this.depth = depth;

        this.inputSize = width*height*depth;
    }
}
