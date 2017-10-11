/*
-*- coding: utf-8 -*-
Copyright (c) 2017 - for information on the respective copyright owner
see the NOTICE file and/or the repository https://github.com/VolkerFischer/statestream

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



/*
 * Converts the source into a specified color code.
 */
void cgraphics_colorcode(double* source,
                         int w,
                         int h,
                         float* r,
                         float* g,
                         float* b,
                         int colorcorrect);



/*
 * Compute forces between neuron-pools.
 */
void cgraphics_np_force(double* item_pos_X,
                        double* item_pos_Y,
                        int* conn_mat,
                        int items,
                        int min_dist,
                        int max_dist,
                        double* force_X,
                        double* force_Y);



/*
 * Compute distances between tensors.
 */
void cgraphics_tensor_dist(float* x,
                           float* y,
                           float* d,
                           int a,
                           int fx,
                           int fy,
                           int sx,
                           int sy,
                           int m);

