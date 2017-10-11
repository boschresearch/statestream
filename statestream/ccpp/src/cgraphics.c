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



#include <math.h>
#include <stdio.h>
#include <stdlib.h>



void cgraphics_colorcode(double* dat,
                         int w,
                         int h,
                         float* r,
                         float* g,
                         float* b,
                         int colorcorrect)
{
    // counter
    double* c;
    long int EE = 256 * 256;
    long int L = (long int)w * (long int)h;
    long int l;
    // determine min / max
    double minimum = dat[0];
    double maximum = dat[0];
    int i;
    for (c = dat, l = 0; l < L; c++, l++)
        if (*c < minimum)
            minimum = *c;
        else if (*c > maximum)
            maximum = *c;
    // set color
    if (minimum == maximum)
    {
        for (c = dat, l = 0; l < L; c++, l++)
            *c = 0.0;
    }
    else
    {
        for (c = dat, l = 0; l < L; c++, l++)
        {
            // compute index
            i = (int)(255.0 * (*c - minimum) / (maximum - minimum));
            // color correction
            if (colorcorrect == 1)
                *c = (double)((int)(256 * r[i]) + 256 * (int)(256 * g[i]) + EE * (int)(256 * b[i] * 0.5));
            else
                *c = (double)((int)(256 * b[i]) + 256 * (int)(256 * g[i]) + EE * (int)(256 * r[i]));
        }
    }
}





void cgraphics_np_force(double* item_pos_X,
                        double* item_pos_Y,
                        int* conn_mat,
                        int items,
                        int min_dist,
                        int max_dist,
                        double* force_X,
                        double* force_Y)
{
    // counter
    long int i;
    long int j;
    int src_np;
    int tgt_np;
    double* pxi;
    double* pyi;
    double* pxj;
    double* pyj;
    double* fxi;
    double* fyi;
    double* fxj;
    double* fyj;

    double diffx;
    double diffy;
    double dist;
    double dist3;

    for (pxi=item_pos_X, pyi=item_pos_Y, fxi=force_X, fyi=force_Y, i=0; i<items; pxi++, pyi++, fxi++, fyi++, i++)
        for (pxj=item_pos_X, pyj=item_pos_Y, fxj=force_X, fyj=force_Y, j=0; j<items; pxj++, pyj++, fxj++, fyj++, j++)
        {
            // Forces: Pairwise repulsive and connected spring.
            if (j < i || conn_mat[j + i * items] == 1)
            {
                diffx = *pxi - *pxj;
                diffy = *pyi - *pyj;
                dist = sqrt(diffx * diffx + diffy * diffy);
                if (j < i && dist < min_dist)
                {
                    if (dist > 2)
                    {
                        dist3 = min_dist / dist;
                        dist3 = 4 * log(dist3 * dist3 * dist3) / dist;
                        *fxi += diffx * dist3;
                        *fxj -= diffx * dist3;
                        *fyi += diffy * dist3;
                        *fyj -= diffy * dist3;
                    }
                    else
                    {
                        dist3 = (double)rand() / (double)RAND_MAX;
                        *fxi += 2 * dist3;
                        *fxj -= 2 * dist3;
                        dist3 = (double)rand() / (double)RAND_MAX;
                        *fyi += 2 * dist3;
                        *fyj -= 2 * dist3;
                    }
                }
                if (conn_mat[j + i * items] == 1)
                {
                    if (dist > max_dist)
                    {
                        dist3 = dist / max_dist;
                        dist3 = 4 * log(dist3 * dist3 * dist3) / dist;
                        *fxi -= diffx * dist3;
                        *fxj += diffx * dist3;
                        *fyi -= diffy * dist3;
                        *fyj += diffy * dist3;
                    }
                }
            }
        }



//    // Update forces for np repulsion.
//            for x0,X0 in self.brain.iteritems():
//                for x1,X1 in self.brain.iteritems():
//                    if x0 != x1:
//                        diff = X0['pos'] - X1['pos']
//                        dist = np.linalg.norm(diff)
//                        if dist < min_dist:
//                            if dist > 2:
//                                X0['force'] += 1000 * diff / (dist**3)
//                            elif dist < 2:
//                                X0['force'] += 2 * np.random.rand(2)
}




/*
    modi:
        -1: inf-norm
        0:  0-norm
        1:  1-norm
        2:  2-norm
        3:  dot-product
        4:  cosine distance
*/
void cgraphics_tensor_dist(float* x,
                 float* y,
                 float* d,
                 int a,
                 int fx,
                 int fy,
                 int sx,
                 int sy,
                 int m)
{
    float* px;
    float* py;

    float dist;
    float normx;
    float normy;

    long int s_cntr;
    long int idx_x;
    long int idx_y;

    long int sxy = sx * sy;

    int a_cntr;
    int fx_cntr;
    int fy_cntr;

    for (a_cntr=0; a_cntr<a; ++a_cntr)
    {
        for (fx_cntr=0; fx_cntr<fx; ++fx_cntr)
        {
            idx_x = fx_cntr * sxy + a_cntr * (fx * sxy);
            for (fy_cntr=0; fy_cntr<fy; ++fy_cntr)
            {
                dist = 0.0;
                idx_y = fy_cntr * sxy + a_cntr * (fy * sxy);
                if (m == -1)
                {
                    for (s_cntr=0, px=x + idx_x, py=y + idx_y; s_cntr<sxy; ++s_cntr, ++px, ++py)
                        if (dist < fabs(*px - *py))
                            dist = abs(*px - *py);
                }
                else if (m == 0)
                {
                    for (s_cntr=0, px=x + idx_x, py=y + idx_y; s_cntr<sxy; ++s_cntr, ++px, ++py)
                        if (*px - *py != 0)
                            dist += 1.0;
                }
                else if (m == 1)
                {
                    for (s_cntr=0, px=x + idx_x, py=y + idx_y; s_cntr<sxy; ++s_cntr, ++px, ++py)
                        if (*px - *py != 0)
                            dist += fabs(*px - *py);
                }
                else if (m == 2)
                {
                    for (s_cntr=0, px=x + idx_x, py=y + idx_y; s_cntr<sxy; ++s_cntr, ++px, ++py)
                        if (*px - *py != 0)
                            dist += (*px - *py) * (*px - *py);
                    if (dist > 1e-12)
                        dist = sqrt(dist);
                    else
                        dist = 0.0;
                }
                else if (m == 3)
                {
                    for (s_cntr=0, px=x + idx_x, py=y + idx_y; s_cntr<sxy; ++s_cntr, ++px, ++py)
                        dist += *px * *py;
                }
                else if (m == 4)
                {
                    normx = 0.0;
                    normy = 0.0;
                    for (s_cntr=0, px=x + idx_x, py=y + idx_y; s_cntr<sxy; ++s_cntr, ++px, ++py)
                    {
                        dist += *px * *py;
                        normx += *px * *px;
                        normy += *py * *py;
                    }
                    if (normx > 1e-8 && normy > 1e-8)
                        dist /= sqrt(normx * normy);
                    else
                        dist *= 0.0;
                }

                d[fx_cntr + fy_cntr * fx] += dist / a;
            }
        }
    }
}

