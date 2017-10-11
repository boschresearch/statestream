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



/* Method used to compute intersection ray / circle:
 *     https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
 * 
 * Method used to compute intersection ray / line-segment:
 *     https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
 */
// fast implementation of simultaneous sin / cos computation
void my_sincos(float a, float* c, float* s) {
#  if defined (__i386__) && !defined (NO_ASM)
#    if defined __GNUC__
#      define ASM_SINCOS
	asm ("fsincos" : "=t" (*s), "=u" (*c) : "0" (a));
#    elif defined _MSC_VER
#      define ASM_SINCOS
		__asm fld a
		__asm fsincos
		__asm fstp *s
		__asm fstp *c
#    endif
#  endif
#  ifndef ASM_SINCOS
	*c = cos(a);
	*s = sin(a);
#  endif
}

/*
 * Compute distance vector from point to line-segment
 */
void my_cdistPLS(float ls_x1,
		 float ls_y1,
		 float ls_x2,
		 float ls_y2,
		 float p_x,
		 float p_y,
		 float* Dx,
		 float* Dy)
{
	float t0;
	float m_x, m_y;
	float bp_x, bp_y;
	
	// compute direction of line segment
	m_x = ls_x2 - ls_x1;
	m_y = ls_y2 - ls_y1;

	// compute direction from point to line segment corner
	bp_x = p_x - ls_x1;
	bp_y = p_y - ls_y1;
	
	// compute projection of p onto line segment
	if (fabs(m_x) < 1e-6 && fabs(m_y) < 1e-6)
		t0 = 0.0;
	else
		t0 =  (m_x * bp_x + m_y * bp_y) / (m_x * m_x + m_y * m_y);
	
	// dependent on t0 compute distance vector
	if (t0 <= 0.0)
	{
		*Dx = bp_x;
		*Dy = bp_y;
	}
	else if (t0 >= 1.0)
	{
		*Dx = bp_x - m_x;
		*Dy = bp_y - m_y;
	}
	else
	{
		*Dx = bp_x - t0 * m_x;
		*Dy = bp_y - t0 * m_y;
	}
}  


/* 
 * World update
 */
void csim(float* world_params,
	     float* ls_x1, float* ls_y1, float* ls_x2, float* ls_y2,
	     float* ls_R, float* ls_G, float* ls_B,
	     float* m_x1, float* m_y1, float* m_x2, float* m_y2, float* c_x,
	     float* c_y, float* c_r, float* c_a,
	     float* c_dx, float* c_dy, float* c_da,
	     float* c_Fx, float* c_Fy, float* c_Fa,
	     float* c_R,  float* c_G, float* c_B,
	     float* a_x,  float* a_y, float* a_a, float* a_r,
	     float* a_dx, float* a_dy, float* a_da,
	     float* a_Fx, float* a_Fy, float* a_Fa,
	     float* a_R, float* a_G, float* a_B,
	     float* a_lookat, float* a_fF, float* a_pF,
	     float* a_motor,
	     int no_ls,
	     int no_m,
	     int no_c,
	     int no_a,
	     int no_f,
	     int no_p,
	     int no_as,
	     int no_ss,
	     float* a_ddx, float* a_ddy, float* a_dda,
	     float* a_sensor,
	     float* f_R,
	     float* f_G,
	     float* f_B,
	     float* f_D,
	     float* p_R,
	     float* p_G,
	     float* p_B,
	     float* p_D)
{
	
	// declare counters
	int ls;
	int m;
	int c;
	int a, A;
	int r;
	int no_rays;
	int modus;
	
	// declare pointers
	float* P_ls_x1;
	float* P_ls_y1;
	float* P_ls_x2;
	float* P_ls_y2;
	float* P_c_x;
	float* P_c_y;
	float* P_c_r;
	float* P_m_x1;
	float* P_m_y1;
	float* P_m_x2;
	float* P_m_y2;
	float* P_a_x;
	float* P_a_y;
	float* P_a_a;
	float* P_a_r;
	float* P_A_x;
	float* P_A_y;
	float* P_A_r;
	float* P_a_lookat;
	float* P_a_fF;
	float* P_a_pF;
	float* P_R;
	float* P_G;
	float* P_B;
	float* P_D;
	float* P_T_R;
	float* P_T_G;
	float* P_T_B;
	
	// some variables needed for computation
	float val1;
	float val2;
	float min_dist;
	float dist;
	float coord;
	float det;
	float angle;
	int target_id;
	int target_type;			// 0=ls, 1=m, 2=c, 3=a
	float target_coord;
	float target_x;
	float target_y;
	float x1, y1;
	float x2, y2;
	float x3, y3;
	float ray_angle;
	float ray_x[1], ray_y[1];
	float iter;

	float pi = 3.1415926;
	float pi2 = 2 * pi;
	
	float EYE_SIZE = 0.1;
	float IRIS_SIZE = 0.02;
	
	// world parameter
	float friction = world_params[0];
	float vel_MM_trans = world_params[1];
	float vel_MM_rot = world_params[2];
	float rep = world_params[3];

	// test sin / cos function
/*
	my_sincos(0.0, ray_x, ray_y);
	printf ("cos(0) = %f, sin(0) = %f\n", *ray_x, *ray_y);
	my_sincos(0.25*3.1415926, ray_x, ray_y);
	printf ("cos(piv) = %f, sin(piv) = %f\n", *ray_x, *ray_y);
	my_sincos(0.5*3.1415926, ray_x, ray_y);
	printf ("cos(pih) = %f, sin(pih) = %f\n", *ray_x, *ray_y);
	my_sincos(3.1415926, ray_x, ray_y);
	printf ("cos(pi) = %f, sin(pi) = %f\n", *ray_x, *ray_y);
*/


	// update forces onto agents
	// P_a is pose
	// P_c is force
	// loop over agents pose and force
	for (a=0, P_a_x=a_x, P_a_y=a_y, P_c_x=a_Fx, P_c_y=a_Fy;
		a<no_a;
		a++, P_a_x++, P_a_y++, P_c_x++, P_c_y++)
	{
		// loop over agents - repulsion agent vs agent
		// P_A is pose
		for (c=0, P_A_x=a_x, P_A_y=a_y;
			c<no_a;
			c++, P_A_x++, P_A_y++)
		{
			if (c != a)
			{
				// compute difference
				x1 = *P_a_x - *P_A_x;
				y1 = *P_a_y - *P_A_y;
				// compute distance
				dist = sqrt(x1*x1 + y1*y1);
				if (dist < 1e-4)
					dist = 1e-4;
				// update force if too near
				if (dist < rep)
				{
					*P_c_x += x1 * (rep - dist) * (rep - dist) / dist;
					*P_c_y += y1 * (rep - dist) * (rep - dist) / dist;
				}
			}
		}
		// loop over circles - repulsion agent vs circle
		// P_A is pose
		for (c=0, P_A_x=c_x, P_A_y=c_y;
			c<no_c;
			c++, P_A_x++, P_A_y++)
		{
			// compute difference
			x1 = *P_a_x - *P_A_x;
			y1 = *P_a_y - *P_A_y;
			// compute distance
			dist = sqrt(x1*x1 + y1*y1);
			if (dist < 1e-4)
				dist = 1e-4;
			// update force if too near
			if (dist < rep)
			{
				*P_c_x += x1 * (rep - dist) * (rep - dist) / dist;
				*P_c_y += y1 * (rep - dist) * (rep - dist) / dist;
			}
		}
		// loop over line segments - repulsion agent vs line segment
		for (c=0, P_ls_x1=ls_x1, P_ls_y1=ls_y1, P_ls_x2=ls_x2, P_ls_y2=ls_y2;
			c<no_ls;
			c++, P_ls_x1++, P_ls_y1++, P_ls_x2++, P_ls_y2++)
		{
			// compute distance vector of agent to line-segment
			my_cdistPLS(*P_ls_x1, *P_ls_y1, *P_ls_x2, *P_ls_y2, *P_a_x, *P_a_y, ray_x, ray_y);
			// compute dist
			dist = sqrt(*ray_x * *ray_x + *ray_y * *ray_y);
			if (dist < 1e-4)
				dist = 1e-4;
			// update force if too near
			if (dist < rep)
			{
				*P_c_x += *ray_x * (rep - dist) * (rep - dist);
				*P_c_y += *ray_y * (rep - dist) * (rep - dist);
			}
		}
		// loop over mirrors - repulsion agent vs mirror
		for (c=0, P_ls_x1=m_x1, P_ls_y1=m_y1, P_ls_x2=m_x2, P_ls_y2=m_y2;
			c<no_m;
			c++, P_ls_x1++, P_ls_y1++, P_ls_x2++, P_ls_y2++)
		{
			// compute distance vector of agent to line-segment
			my_cdistPLS(*P_ls_x1, *P_ls_y1, *P_ls_x2, *P_ls_y2, *P_a_x, *P_a_y, ray_x, ray_y);
			// compute dist
			dist = sqrt(*ray_x* *ray_x + *ray_y * *ray_y);
			if (dist < 1e-4)
				dist = 1e-4;
			// update force if too near
			if (dist < rep)
			{
				*P_c_x += *ray_x * (rep - dist) * (rep - dist);
				*P_c_y += *ray_y * (rep - dist) * (rep - dist);
			}
		}
	}

	
	
	
	
	
	// update forces onto circles
	// P_a is pose
	// P_c is force
	// loop over circles pose and force
	for (a=0, P_a_x=c_x, P_a_y=c_y, P_c_x=c_Fx, P_c_y=c_Fy;
		a<no_c;
		a++, P_a_x++, P_a_y++, P_c_x++, P_c_y++)
	{
		// loop over agents - repulsion circle vs agent
		// P_A is pose
		for (c=0, P_A_x=a_x, P_A_y=a_y;
			c<no_a;
			c++, P_A_x++, P_A_y++)
		{
			// compute difference
			x1 = *P_a_x - *P_A_x;
			y1 = *P_a_y - *P_A_y;
			// compute distance
			dist = sqrt(x1*x1 + y1*y1);
			if (dist < 1e-4)
				dist = 1e-4;
			// update force if too near
			if (dist < rep)
			{
				*P_c_x += x1 * (rep - dist) * (rep - dist) / dist;
				*P_c_y += y1 * (rep - dist) * (rep - dist) / dist;
			}
		}
		// loop over circles - repulsion circle vs circle
		// P_A is pose
		for (c=0, P_A_x=c_x, P_A_y=c_y;
			c<no_c;
			c++, P_A_x++, P_A_y++)
		{
			if (c != a)
			{
				// compute difference
				x1 = *P_a_x - *P_A_x;
				y1 = *P_a_y - *P_A_y;
				// compute distance
				dist = sqrt(x1*x1 + y1*y1);
				if (dist < 1e-4)
					dist = 1e-4;
				// update force if too near
				if (dist < rep)
				{
					*P_c_x += x1 * (rep - dist) * (rep - dist) / dist;
					*P_c_y += y1 * (rep - dist) * (rep - dist) / dist;
				}
			}
		}
		// loop over line segments - repulsion circle vs line segment
		for (c=0, P_ls_x1=ls_x1, P_ls_y1=ls_y1, P_ls_x2=ls_x2, P_ls_y2=ls_y2;
			c<no_ls;
			c++, P_ls_x1++, P_ls_y1++, P_ls_x2++, P_ls_y2++)
		{
			// compute distance vector of agent to line-segment
			my_cdistPLS(*P_ls_x1, *P_ls_y1, *P_ls_x2, *P_ls_y2, *P_a_x, *P_a_y, ray_x, ray_y);
			// compute dist
			dist = sqrt(*ray_x* * ray_x + *ray_y * *ray_y);
			if (dist < 1e-4)
				dist = 1e-4;
			// update force if too near
			if (dist < rep)
			{
				*P_c_x += *ray_x * (rep - dist) * (rep - dist);
				*P_c_y += *ray_y * (rep - dist) * (rep - dist);
			}
		}
		// loop over mirrors - repulsion circle vs mirror
		for (c=0, P_ls_x1=m_x1, P_ls_y1=m_y1, P_ls_x2=m_x2, P_ls_y2=m_y2;
			c<no_m;
			c++, P_ls_x1++, P_ls_y1++, P_ls_x2++, P_ls_y2++)
		{
			// compute distance vector of agent to line-segment
			my_cdistPLS(*P_ls_x1, *P_ls_y1, *P_ls_x2, *P_ls_y2, *P_a_x, *P_a_y, ray_x, ray_y);
			// compute dist
			dist = sqrt(*ray_x* *ray_x + *ray_y * *ray_y);
			if (dist < 1e-4)
				dist = 1e-4;
			// update force if too near
			if (dist < rep)
			{
				*P_c_x += *ray_x * (rep - dist) * (rep - dist);
				*P_c_y += *ray_y * (rep - dist) * (rep - dist);
			}
		}
	}

	
	
	
	
	
	// update cirle's dynamics
	// P_a is pose
	// P_A is velocity
	// P_c is force
	for (c=0, P_a_x=c_x, P_a_y=c_y, P_a_r=c_a, P_A_x=c_dx, P_A_y=c_dy, P_A_r=c_da, P_c_x=c_Fx, P_c_y=c_Fy, P_c_r=c_Fa;
		c<no_c;
		c++, P_a_x++, P_a_y++, P_a_r++, P_A_x++, P_A_y++, P_A_r++, P_c_x++, P_c_y++, P_c_r++)
	{
		// apply friction to velocity
		*P_A_x *= friction;
		*P_A_y *= friction;
		*P_A_r *= friction;
		
		// update velocity with force (forces are already in global frame)
		*P_A_x += *P_c_x;
		*P_A_y += *P_c_y;
		*P_A_r += *P_c_r;
		
		// clip velocities
		if (*P_A_x < -vel_MM_trans) *P_A_x = -vel_MM_trans;
		else if (*P_A_x > vel_MM_trans) *P_A_x = vel_MM_trans;
		if (*P_A_y < -vel_MM_trans) *P_A_y = -vel_MM_trans;
		else if (*P_A_y > vel_MM_trans) *P_A_y = vel_MM_trans;
		if (*P_A_r < -vel_MM_rot) *P_A_r = -vel_MM_rot;
		else if (*P_A_r > vel_MM_rot) *P_A_r = vel_MM_rot;
		
		// update pose with velocity (both in global system)
		*P_a_x += *P_A_x;
		*P_a_y += *P_A_y;
		*P_a_r += *P_A_r;	// this is the angular update
	}

	
	
	
	
	
	// update agent's dynamics
	// P_a is pose
	// P_A is velocity
	// P_c is force
	for (a=0, P_a_x=a_x, P_a_y=a_y, P_a_a=a_a, P_A_x=a_dx, P_A_y=a_dy, P_A_r=a_da, P_c_x=a_Fx, P_c_y=a_Fy, P_c_r=a_Fa;
		a<no_a;
		a++, P_a_x++, P_a_y++, P_a_a++, P_A_x++, P_A_y++, P_A_r++, P_c_x++, P_c_y++, P_c_r++)
	{
		// apply friction to velocity
		*P_A_x *= friction;
		*P_A_y *= friction;
		*P_A_r *= friction;

		// update velocity with force (forces are already in global frame)
		*P_A_x += *P_c_x;
		*P_A_y += *P_c_y;
		*P_A_r += *P_c_r;
		
		// clip velocities
		if (*P_A_x < -vel_MM_trans) *P_A_x = -vel_MM_trans;
		else if (*P_A_x > vel_MM_trans) *P_A_x = vel_MM_trans;
		if (*P_A_y < -vel_MM_trans) *P_A_y = -vel_MM_trans;
		else if (*P_A_y > vel_MM_trans) *P_A_y = vel_MM_trans;
		if (*P_A_r < -vel_MM_rot) *P_A_r = -vel_MM_rot;
		else if (*P_A_r > vel_MM_rot) *P_A_r = vel_MM_rot;

		// update pose with velocity (both in global system)
		*P_a_x += *P_A_x;
		*P_a_y += *P_A_y;
		*P_a_a += *P_A_r;	// this is the angular update
	}


	// loop over fov / per modus
	for (modus=0; modus<2; modus++)
	{	
		// loop over all agents
		for (a=0, P_a_x=a_x, P_a_y=a_y, P_a_a=a_a, P_a_r=a_r, P_a_lookat=a_lookat, P_a_fF=a_fF, P_a_pF=a_pF;
		     a<no_a;
		     a++, P_a_x++, P_a_y++, P_a_a++, P_a_r++, P_a_lookat++, P_a_pF++, P_a_fF++)
		{
		
			// dependent on modes set agent dependent variables
			if (modus == 0)
			{
				iter = 2.0 * *P_a_fF / no_f;
				P_R = &f_R[a * no_f];
				P_G = &f_G[a * no_f];
				P_B = &f_B[a * no_f];
				P_D = &f_D[a * no_f];
				no_rays = no_f;
			}
			else 
			{
				iter = 2.0 * *P_a_pF / no_p;
				P_R = &p_R[a * no_p];
				P_G = &p_G[a * no_p];
				P_B = &p_B[a * no_p];
				P_D = &p_D[a * no_p];
				no_rays = no_p;
			}
			
			// loop over all rays
			for (r=0; r<no_rays; r++, P_R++, P_G++, P_B++, P_D++)
			{
				// begin with min dist >>
				min_dist = 1e+6;
				target_type = -1;
				
				// compute global angle of ray
				if (modus == 0)
				  ray_angle = *P_a_a + *P_a_lookat - *P_a_fF + r * iter;
				else
				  ray_angle = *P_a_a + *P_a_lookat - *P_a_pF + r * iter;
				
				// normalize angle
				if (ray_angle < 0.0)
					ray_angle += pi2;
				else if (ray_angle > pi2)
					ray_angle -= pi2;
				
				// compute unit direction of ray
				my_sincos(ray_angle, ray_x, ray_y);
				
				// loop over all circles
				for (c=0, P_c_x=c_x, P_c_y=c_y, P_c_r=c_r;
					c<no_c;
					c++, P_c_x++, P_c_y++, P_c_r++)
				{

					// compute distance from circle to ray origin
					x1 = *P_a_x - *P_c_x;
					y1 = *P_a_y - *P_c_y;
					
					// compute number needed several times
					val1 = *ray_x * x1 + *ray_y * y1;
					
					// compute determinant
					det = val1*val1 - x1*x1 - y1*y1 + (*P_c_r)*(*P_c_r);
					
					// case specific distance
					if (det > 0.0)
					{
						det = sqrt(det);
						if (-val1 - det > 0.0)
							dist = -val1 - det;
						else
							if (-val1 + det > 0.0)
								dist = -val1 + det;
							else
								continue;
					}
					else
						continue;
					
					// update minimal distance
					if (dist < min_dist)
					{
						min_dist = dist;
						target_id = c;
						target_type = 2;
					}
				}
				
				// loop over all agents
				for (A=0, P_A_x=a_x, P_A_y=a_y, P_A_r=a_r;
					A<no_a;
					A++, P_A_x++, P_A_y++, P_A_r++)
				{
					
					// self not visible
					if (A == a)
						continue;

					// compute distance from circle to ray origin
					x1 = *P_a_x - *P_A_x;
					y1 = *P_a_y - *P_A_y;
					
					// compute number needed several times
					val1 = *ray_x * x1 + *ray_y * y1;
					
					// compute determinant
					det = val1*val1 - x1*x1 - y1*y1 + (*P_A_r)*(*P_A_r);
					
					// case specific distance
					if (det > 0.0)
					{
						det = sqrt(det);
						if (-val1 - det > 0.0)
							dist = -val1 - det;
						else
							if (-val1 + det > 0.0)
								dist = -val1 + det;
							else
								continue;
					}
					else
						continue;
					
					// update minimal distance
					if (dist < min_dist)
					{
						min_dist = dist;
						target_id = A;
						target_type = 3;
					}
				}
				
				// loop over all line segments
				for (ls=0, P_ls_x1=ls_x1, P_ls_y1=ls_y1, P_ls_x2=ls_x2, P_ls_y2=ls_y2;
				     ls<no_ls;
				     ls++, P_ls_x1++, P_ls_y1++, P_ls_x2++, P_ls_y2++) {

					// compute vector differences
					x1 = *P_a_x - *P_ls_x1;
					y1 = *P_a_y - *P_ls_y1;
					x2 = *P_ls_x2 - *P_ls_x1;
					y2 = *P_ls_y2 - *P_ls_y1;

					val2 = x2*y1 - y2*x1;
					if (val2 >= 0)
					{
						x3 = - *ray_y;
						y3 = *ray_x;
					}
					else
					{
						x3 = *ray_y;
						y3 = - *ray_x;
					}
					
					// this is needed two times
					val1 = x2*x3 + y2*y3;
					
					// check for parallelism (near zero) and wrong side (< 0)
					if (val1 < 1e-6)
						continue;
					
					// compute local coord on line segment
					coord = (x1*x3 + y1*y3) / val1;
					// virtual intersection
					if (coord < 0.0 || coord > 1.0)
						continue;
					
					// compute distance
					dist = fabs(val2) / val1;
					
					// update minimal distance
					if (dist < min_dist)
					{
						min_dist = dist;
						target_type = 0;
						target_id = ls;
						target_coord = coord;
						target_x = x2;
						target_y = y2;
					}
				}
				
				// loop over all mirrors
				for (m=0, P_m_x1=m_x1, P_m_y1=m_y1, P_m_x2=m_x2, P_m_y2=m_y2;
				     m<no_m;
				     m++, P_m_x1++, P_m_y1++, P_m_x2++, P_m_y2++) {

					// compute vector differences
					x1 = *P_a_x - *P_m_x1;
					y1 = *P_a_y - *P_m_y1;
					x2 = *P_m_x2 - *P_m_x1;
					y2 = *P_m_y2 - *P_m_y1;

					val2 = x2*y1 - y2*x1;
					if (val2 >= 0)
					{
						x3 = - *ray_y;
						y3 = *ray_x;
					}
					else
					{
						x3 = *ray_y;
						y3 = - *ray_x;
					}
					
					// this is needed two times
					val1 = x2*x3 + y2*y3;
					
					// check for parallelism (near zero) and wrong side (< 0)
					if (val1 < 1e-6)
						continue;
					
					// compute local coord on line segment
					coord = (x1*x3 + y1*y3) / val1;
					// virtual intersection
					if (coord < 0.0 || coord > 1.0)
						continue;
					
					// compute distance
					dist = fabs(val2) / val1;
					
					// update minimal distance
					if (dist < min_dist) {
						min_dist = dist;
						target_id = m;
						target_type = 1;
					}
				}
				
				// for mirrors compute reflections
				if (target_type == 1) {
					// origin of reflected ray
					x1 = min_dist * *ray_x;
					y1 = min_dist * *ray_y;

					// angle of mirror
					x2 = P_m_x1[target_id] - P_m_x2[target_id];
					y2 = P_m_y1[target_id] - P_m_y2[target_id];
					angle = atan2(y2, x2);
					
					// angle of reflected ray
					angle -= ray_angle;
					
					// loop over all line segments
					
					// loop over all circles
					
					// loop over ALL agents
				}
				
				// dependent on target type compute texture
				// line segment
				if (target_type == 0) {
					// get pointer to tex parameters
					P_T_R = &ls_R[4 * target_id];
					P_T_G = &ls_G[4 * target_id];
					P_T_B = &ls_B[4 * target_id];

					// transform local into global tex-coord
					target_coord *= sqrt(target_x * target_x + target_y * target_y);
					
					// compute phase and final R value
					val2 = 1.0 / (P_T_R[2] + P_T_R[3]);
					//val1 = nur_nachkomma_von(target_coord * val2);
					val1 = 0.0;
					if (val1 > P_T_R[2] * val2)
						*P_R = P_T_R[1];
					else 
						*P_R = P_T_R[0];

					// compute phase and final G value
					val2 = 1.0 / (P_T_G[2] + P_T_G[3]);
					//val1 = nur_nachkomma_von(target_coord * val2);
					val1 = 0.0;
					if (val1 > P_T_G[2] * val2)
						*P_G = P_T_G[1];
					else
						*P_G = P_T_G[0];

					// compute phase and final G value
					val2 = 1.0 / (P_T_B[2] + P_T_B[3]);
					//val1 = nur_nachkomma_von(target_coord * val2);
					val1 = 0.0;
					if (val1 > P_T_B[2] * val2)
						*P_B = P_T_B[1];
					else
						*P_B = P_T_B[0];
				}

				// circle
				else if (target_type == 2) {
					// get pointer to tex parameters
					P_T_R = &c_R[4 * target_id];
					P_T_G = &c_G[4 * target_id];
					P_T_B = &c_B[4 * target_id];

					// compute target point with circle as origin
					x1 = min_dist * *ray_x - c_x[target_id];
					y1 = min_dist * *ray_y - c_y[target_id];
					
					// angle of target point with circle as origin
					angle = atan2(y1, x1);
					
					// consider radius
					angle *= 2.0 * c_r[target_id];
					
					// compute phase and final R value
					val2 = 1.0 / (P_T_R[2] + P_T_R[3]);
					//val1 = nur_nachkomma_von(angle * val2);
					val1 = 0.0;
					if (val1 > P_T_R[2] * val2)
						*P_R = P_T_R[1];
					else
						*P_R = P_T_R[0];

					// compute phase and final G value
					val2 = 1.0 / (P_T_G[2] + P_T_G[3]);
					//val1 = nur_nachkomma_von(angle * val2);
					val1 = 0.0;
					if (val1 > P_T_G[2] * val2)
						*P_G = P_T_G[1];
					else
						*P_G = P_T_G[0];

					// compute phase and final B value
					val2 = 1.0 / (P_T_B[2] + P_T_B[3]);
					//val1 = nur_nachkomma_von(angle * val2);
					val1 = 0.0;
					if (val1 > P_T_B[2] * val2)
						*P_B = P_T_B[1];
					else
						*P_B = P_T_B[0];
				}
				
				// agent
				else if (target_type == 3) {
					// get pointer to tex parameters
					P_T_R = &a_R[4 * target_id];
					P_T_G = &a_G[4 * target_id];
					P_T_B = &a_B[4 * target_id];

					// compute target point with circle as origin
					x1 = min_dist * *ray_x - a_x[target_id];
					y1 = min_dist * *ray_y - a_y[target_id];
					
					// angle of target point with circle as origin
					// considering agent's orientation
					angle = atan2(y1, x1) - a_a[target_id];

					// consider agent's eye
					if (fabs(angle) < EYE_SIZE) {
						// consider iris
						if (fabs(angle - a_lookat[target_id]) < IRIS_SIZE) {
							*P_R = 0.0; *P_G = 0.0; *P_B = 0.0;
						}
						// eye
						else {
							*P_R = 1.0; *P_G = 1.0; *P_B = 1.0;
						}
					}
					else {
						// consider radius
						angle *= 2.0 * a_r[target_id];
						
						// compute phase and final R value
						val2 = 1.0 / (P_T_R[2] + P_T_R[3]);
						//val1 = nur_nachkomma_von(angle * val2);
						val1 = 0.0;
						if (val1 > P_T_R[2] * val2)
							*P_R = P_T_R[1];
						else
							*P_R = P_T_R[0];

						// compute phase and final G value
						val2 = 1.0 / (P_T_G[2] + P_T_G[3]);
						//val1 = nur_nachkomma_von(angle * val2);
						val1 = 0.0;
						if (val1 > P_T_G[2] * val2)
							*P_G = P_T_G[1];
						else
							*P_G = P_T_G[0];

						// compute phase and final B value
						val2 = 1.0 / (P_T_B[2] + P_T_B[3]);
						//val1 = nur_nachkomma_von(angle * val2);
						val1 = 0.0;
						if (val1 > P_T_B[2] * val2)
							*P_B = P_T_B[1];
						else
							*P_B = P_T_B[0];
					}
				}
				
				// finally set ray distance
				*P_D = min_dist;
			}
		}
	}

	// compute agent's haptic / sensori input
	// determine angular iter between sensors (same for all agents)
	iter = pi2 / (float)no_ss;
	for (a=0; a<no_a; a++)
	{
		// point to first peripherie distance
		P_D = &p_D[a * no_p];
		// loop over haptic sensors
		// angle is in the lookat frame
		for (r=0, P_A_r=&a_sensor[a * no_ss], angle=-pi-a_lookat[a]; 
			 r<no_ss;
			 r++, P_A_r++, angle+=iter)
		{
			*P_A_r = 0.0;
			// convert angle to index (everything in lookat dist frame)
			c = ((int)(no_p * angle / pi2) + no_p) % no_p;
			val1 = (2.0 * rep - P_D[c]) / (2.0 * rep);
			// check distance (val1 is in [0, 1])
			if (val1 > 0.0)
				*P_A_r = -log(1.0 - val1);
		}
	}
}