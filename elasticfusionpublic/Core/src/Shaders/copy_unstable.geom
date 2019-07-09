/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#version 400 core
#extension GL_ARB_gpu_shader5 : enable

layout(points) in;

in vec4 vPosition[];
in vec4 vColor[];
in vec4 vNormRad[];
flat in int test[];
flat in int vertexId[];

layout(points,max_vertices = 1,stream = 0) out;
out vec4 vPosition0;
out vec4 vColor0;
out vec4 vNormRad0;

layout(points,max_vertices = 1,stream = 1) out;
flat out int deleted_id;

uniform int isNew;

void main() 
{
    if(isNew == 0 && test[0] > 0)
    {
        deleted_id = vertexId[0];
        EmitStreamVertex(1);
    }
    else if(test[0] > 0)
    {
        vPosition0 = vPosition[0];
        vColor0 = vColor[0];
        vNormRad0 = vNormRad[0];
        EmitStreamVertex(0);
    }
}
