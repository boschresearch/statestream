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



/* 
 * 
 * Inputs:
 * 	world_params	[3]		friction, vel_MM_trans, vel_MM_rot, rep
 * 
 * 	ls_x1 		[no_ls]		line segments
 * 	ls_y1 		[no_ls]
 * 	ls_x2 		[no_ls]
 * 	ls_y2 		[no_ls]
 * 	ls_R		[4 * no_ls]	min, max, width min, width max
 * 	ls_G		[4 * no_ls]
 * 	ls_B		[4 * no_ls]
 * 
 * 	m_x1 		[no_m]		mirrors
 * 	m_y1 		[no_m]
 * 	m_x2 		[no_m]
 * 	m_y2 		[no_m]
 * 
 * 	c_x 		[no_c]		circles (non-agents)
 * 	c_y 		[no_c]
 * 	c_r 		[no_c]
 * 	c_dx		[no_c]		velocity of circles
 * 	c_dy		[no_c]
 * 	c_da		[no_c]
 * 	c_R		[4 * no_c]	min, max, width min, width max
 * 	c_G		[4 * no_c]
 * 	c_B		[4 * no_c]
 
 * 	a_x 		[no_a]		agents
 * 	a_y 		[no_a]
 * 	a_r 		[no_a]
 *	a_dx		[no_a]		agents velocity (global=world system)
 *	a_dy		[no_a]
 *	a_da		[no_a] 
 *	a_Fx		[no_a]		force upon agents (also already in global frame)
 *	a_Fy		[no_a]
 *	a_Fa		[no_a] 
 * 	a_R 		[no_a]		agent's color
 * 	a_G 		[no_a]
 * 	a_B 		[no_a]
 * 	a_fa 		[no_a]		fov middle angle
 * 	a_pa		[no_a]		per middle angle
 * 	a_fF		[no_a]		fov field of view
 * 	a_pF		[no_a]		per field of view
 *	a_a		[no_a * no_as]	action signals for agents
 * 	no_ls
 * 	no_m			number of mirrors
 * 	no_c			number of circles
 * 	no_a			number of agents
 * 	no_f			number of rays in fovea
 * 	no_p			number of rays in periphery
 *	no_as			number of agent action-segments
 *	no_ss			number of agent sensor-segments
 * 
 * Output:
 *	a_ddx		[no_a]		actual acceleration of agents
 *	a_ddy		[no_a]
 *	a_dda		[no_a]
 *	a_s		[no_a * no_ss]	sensor signals of agents
 * 	f_R 		[no_a * no_f]
 * 	f_G 		[no_a * no_f]
 * 	f_B 		[no_a * no_f]
 * 	f_D 		[no_a * no_f]
 * 	p_R 		[no_a * no_p]
 * 	p_G 		[no_a * no_p]
 * 	p_B 		[no_a * no_p]
 * 	p_D 		[no_a * no_p]
 *
 * 
 * Color code (4 params for each channel):
 * 	[0] :: min val [0, 1]
 *	[1] :: max val [0, 1]
 *	[2] :: width of min val [0, inf)
 *	[3] :: width of max val [0, inf)
 *
 */



// void my_csim(float* in_array, float* out_array, int size);



/*
 * TODO
 * 	normalize angles in -pi ... pi (using global variable PI and new 'norm' function)
 * 	mirror reflections
 * 	integrate motion model for agents and circles
 * 	integrate collision detection + handling
 * 
 * 	integrate action-sensor for agents (also as variables -> add additional sensor segments)
 * 	integrate actions for agents (as texture change)
 * 	integrate orientation for circles (also as variables)
 * 	fast sqrt()
 */



// fast parallel sin/cos computation
void my_sincos(float a, float* s, float* c);



// compute distance point to line-segment
void my_cdistPLS(float ls_x1,
		 float ls_y1,
		 float ls_x2,
		 float ls_y2,
		 float p_x,
		 float p_y,
		 float* dist);



// main simulation c function (see above)
void csim(float* world_params,
	     float* ls_x1,
	     float* ls_y1,
	     float* ls_x2,
	     float* ls_y2,
	     float* ls_R,
	     float* ls_G,
	     float* ls_B,
	     float* m_x1,
	     float* m_y1,
	     float* m_x2,
	     float* m_y2,
	     float* c_x,
	     float* c_y,
	     float* c_r,
	     float* c_a,
	     float* c_dx,
	     float* c_dy,
	     float* c_da,
	     float* c_Fx,
	     float* c_Fy,
	     float* c_Fa,
	     float* c_R,
	     float* c_G,
	     float* c_B,
	     float* a_x,
	     float* a_y,
	     float* a_a,
	     float* a_r,
	     float* a_dx,
	     float* a_dy,
	     float* a_da,
	     float* a_Fx,
	     float* a_Fy,
	     float* a_Fa,
	     float* a_R,
	     float* a_G,
	     float* a_B,
	     float* a_lookat,
	     float* a_fF,
	     float* a_pF,
	     float* a_motor,
	     int no_ls,
	     int no_m,
	     int no_c,
	     int no_a,
	     int no_f,
	     int no_p,
	     int no_as,
	     int no_ss,
	     float* a_ddx,
	     float* a_ddy,
	     float* a_dda,
	     float* a_sensor,
	     float* f_R,
	     float* f_G,
	     float* f_B,
	     float* f_D,
	     float* p_R,
	     float* p_G,
	     float* p_B,
	     float* p_D);
