/*
-*- coding: utf-8 -*-
Copyright (c) 2017 - for information on the respective copyright owner
see the NOTICE file and/or the repository https://github.com/boschresearch/statestream

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/



#include <math.h>
#include <stdio.h>
#include <stdlib.h>



/*
 * Some functions to enable hebbian and hebbian-like plasticities.
 */
void chebb_raw(float* src,
                float* tgt,
                float* upd,
                float tgt_eps,
                int agents,
                int src_X,
                int src_Y,
                int src_C,
                int tgt_X,
                int tgt_Y,
                int tgt_C,
                int rf_X,
                int rf_Y,
                int dil_X,
                int dil_Y,
                int ignore_border)
{
    /*
     * We assume a all arrays to be flattened in row-major style.
     * We also assume upd = [tgt_c, src_c, rf_X, rf_Y].
    */
    // Some counter and pointer counter.
    long int I;
    long int J;

    int a;
    int src_x;
    int src_y;
    int src_c;
    int tgt_x;
    int tgt_y;
    int tgt_c;
    int rf_x;
    int rf_y;

    int rf_Xh = rf_X / 2;
    int rf_Yh = rf_Y / 2;

    int stride_x = src_X / tgt_X;
    int stride_y = src_Y / tgt_Y;
    int border_x = rf_Xh * dil_X / stride_x;
    int border_y = rf_Yh * dil_Y / stride_y;

    float* ptr_f1;
    float* ptr_f2;

    // Loop over target neurons.
    ptr_f1 = tgt;
    for (a=0; a<agents; ++a)
        for (tgt_c=0; tgt_c<tgt_C; ++tgt_c)
            for (tgt_x=0; tgt_x<tgt_X; ++tgt_x)
                for (tgt_y=0; tgt_y<tgt_Y; ++tgt_y)
                {
                    // Check if activation exceeds eps.
                    if (fabs(*ptr_f1) > tgt_eps)
                    {
                        // Check if to ignore borders.
                        if (ignore_border == 1)
                        {
                            // Check if inside borders.
                            if (tgt_x < border_x || tgt_x > tgt_X - border_x || tgt_y < border_y || tgt_y > tgt_Y - border_y)
                                continue;
                            else
                            {
                                // Loop over receptive field.
                                ptr_f2 = upd + tgt_c * tgt_X * tgt_Y * src_C;
                                I = a * src_C * src_X * src_Y + stride_x * tgt_x * src_Y + stride_y * tgt_y;
                                for (src_c=0; src_c<src_C; ++src_c)
                                {
                                    for (rf_x=-rf_Xh; rf_x<=rf_Xh; ++rf_x)
                                    {
                                        // Compute source neuron index (local rf_x).
                                        J = I + rf_x * dil_X * src_Y;
                                        for (rf_y=-rf_Yh; rf_y<=rf_Yh; ++rf_y)
                                        {
                                            // Compute source neuron index (local rf_y).
                                            J += rf_y * dil_Y;
                                            // Update update with hebbian.
                                            *ptr_f2 += *ptr_f1 * src[J];
                                            // Count up pointer in update.
                                            ++ptr_f2;
                                        }
                                    }
                                    // Compute source neuron index (next src channel).
                                    I += src_X * src_Y;
                                }
                            }
                        }
                        else
                        {
                            // Loop over receptive field.
                            ptr_f2 = upd + tgt_c * tgt_X * tgt_Y * src_C;
                            // TODO: do not ignore borders case.
                        }
                    }
                    // Count up pointer over target in any case.
                    ++ptr_f1;
                }
}




