package com.codingapi.deeplearning.demo09.learn.utils;

/**
 * @author lorne
 * @date 2019/12/6
 * @description
 */
public class MaxUtils {

    public static int maxIndex(double[] array){
        int max = 0;
        for(int i=0;i<array.length;i++){
            double val = array[i];
            if(val>=array[max]){
                max = i;
            }
        }
        return max;
    }
}
