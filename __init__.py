from multiprocessing import current_process
import numpy as np

bl_info = {
    "name": "Dream Models",
    "blender": (4, 1, 0),
    "category": "Object",
}

if current_process().name != "__actor__":
    import bpy
    import bmesh
    from .backend import Backend, PointCloudResponse, MeshResponse

    class DreamModelProgress(bpy.types.PropertyGroup):
        active: bpy.props.BoolProperty()
        step: bpy.props.IntProperty()
        steps: bpy.props.IntProperty()

        def draw(self, layout):
            if self.steps == 0:
                layout.progress(
                    text=f'Loading...',
                    factor=0
                )
            else:
                layout.progress(
                    text=f'{self.step} / {self.steps}',
                    factor=self.step / self.steps
                )

    class DreamModelSettings(bpy.types.PropertyGroup):
        image: bpy.props.PointerProperty(type=bpy.types.Image)

        point_cloud_progress: bpy.props.PointerProperty(type=DreamModelProgress)
        mesh_progress: bpy.props.PointerProperty(type=DreamModelProgress)

    class DreamModelPanel(bpy.types.Panel):
        """Creates a Dream Model panel for projection"""
        bl_label = "Dream Model"
        bl_idname = f"DREAM_PT_dream_model_panel"
        bl_category = "Dream"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'

        @classmethod
        def poll(cls, context):
            return True

        def draw(self, context):
            layout = self.layout
            layout.use_property_split = True
            layout.use_property_decorate = False

            layout.template_ID_preview(context.scene.dream_model_settings, "image", new="image.new", open="image.open")

            # point cloud
            operator_row = layout.row(align=True)
            operator_row.scale_y = 1.5
            point_cloud_progress = context.scene.dream_model_settings.point_cloud_progress
            if point_cloud_progress.active:
                point_cloud_progress.draw(operator_row)
                operator_row.operator(DreamModelCancel.bl_idname, text="", icon="X")
            else:
                operator_row.operator(GeneratePointCloud.bl_idname)

            # mesh
            operator_row = layout.row(align=True)
            operator_row.scale_y = 1.5
            mesh_progress = context.scene.dream_model_settings.mesh_progress
            if mesh_progress.active:
                mesh_progress.draw(operator_row)
                operator_row.operator(DreamModelCancel.bl_idname, text="", icon="X")
            else:
                operator_row.operator(GenerateMesh.bl_idname)

    class DreamModelCancel(bpy.types.Operator):
        bl_idname = "dream_models.cancel"
        bl_label = "Cancel Generation"
        bl_description = "Cancel point cloud and mesh generation"
        bl_options = {"REGISTER"}

        @classmethod
        def poll(cls, context):
            return True
        
        def execute(self, context):
            context.scene.dream_model_settings.point_cloud_progress.active = False
            context.scene.dream_model_settings.mesh_progress.active = False
            Backend.shared_close()

            return {'FINISHED'}

    from dream_textures import image_utils

    class GeneratePointCloud(bpy.types.Operator):
        bl_idname = "dream_models.generate_point_cloud"
        bl_label = "Generate Point Cloud"
        bl_description = "Create a point cloud from the input image(s)"
        bl_options = {"REGISTER"}

        @classmethod
        def poll(cls, context):
            return True

        def execute(self, context):
            point_cloud_progress = context.scene.dream_model_settings.point_cloud_progress
            point_cloud_progress.active = True
            point_cloud_progress.step = 0
            point_cloud_progress.steps = 0

            image = image_utils.bpy_to_np(context.scene.dream_model_settings.image)

            mesh = bpy.data.meshes.new(context.scene.dream_model_settings.image.name)

            obj = bpy.data.objects.new(context.scene.dream_model_settings.image.name, mesh)
            context.collection.objects.link(obj)

            modifier = obj.modifiers.new('Point Cloud Preview', 'NODES')
            modifier.node_group = bpy.data.node_groups['Point Cloud Preview']

            obj.select_set(True)
            context.view_layer.objects.active = obj

            backend: Backend = Backend.shared()
            future = backend.point_cloud(image)

            def on_response(future, response: PointCloudResponse):
                mesh.vertices.data.clear_geometry()

                mesh.vertices.add(len(response.vertices))
                mesh.vertices.foreach_set('co', response.vertices.ravel())
                
                color_attribute = mesh.color_attributes.new(
                    name='color',
                    type='FLOAT_COLOR',
                    domain='POINT',
                )
                color_attribute.data.foreach_set('color', response.colors.ravel())

                mesh.update()
                
                point_cloud_progress.step = response.step
                point_cloud_progress.steps = response.steps
            
            def on_done(future):
                point_cloud_progress.active = False
                print('done')
                Backend.shared_close()
            
            def on_exception(_, exception):
                point_cloud_progress.active = False
                print(exception)
                raise exception

            future.add_response_callback(on_response)
            future.add_exception_callback(on_exception)
            future.add_done_callback(on_done)

            return {"FINISHED"}
    
    class GenerateMesh(bpy.types.Operator):
        bl_idname = "dream_models.generate_mesh"
        bl_label = "Generate Mesh"
        bl_description = "Create a mesh from the input image and selected point cloud"
        bl_options = {"REGISTER"}

        @classmethod
        def poll(cls, context):
            return True

        def execute(self, context):
            mesh_progress = context.scene.dream_model_settings.mesh_progress
            mesh_progress.active = True
            mesh_progress.step = 0
            mesh_progress.steps = 0

            image = image_utils.bpy_to_np(context.scene.dream_model_settings.image)
            obj = context.view_layer.objects.active
            point_cloud_mesh = obj.data
            color_attribute = obj.data.color_attributes['color']

            point_cloud = np.zeros((len(point_cloud_mesh.vertices), 3), dtype=np.float32)
            point_cloud_mesh.vertices.foreach_get('co', point_cloud.ravel())
            
            point_cloud_colors = np.zeros((len(color_attribute.data), 4), dtype=np.float32)
            color_attribute.data.foreach_get('color', point_cloud_colors.ravel())
            
            backend: Backend = Backend.shared()
            future = backend.mesh(image, point_cloud, point_cloud_colors)

            def on_response(future, response: MeshResponse):
                return
            
            def on_done(future):
                mesh_progress.active = False
                response: MeshResponse = future.result(last_only=True)
                
                image_texture = image_utils.np_to_bpy(response.texture)

                # build new mesh from result
                mesh = bpy.data.meshes.new(obj.name)
                mesh.from_pydata(response.vertices, [], response.faces)

                uv_layer = mesh.uv_layers.new(name="UVMap")
                uv_layer.data.foreach_set('uv', response.uvs.ravel())

                mesh.update()

                obj.data = mesh
                bpy.data.meshes.remove(point_cloud_mesh) # remove old mesh
                obj.modifiers.remove(obj.modifiers[0]) # remove preview geometry nodes modifier

                # create material
                material = bpy.data.materials.new(name=obj.name)
                material.use_nodes = True
                nodes = material.node_tree.nodes
                links = material.node_tree.links

                nodes.clear()
                
                output_node = nodes.new(type='ShaderNodeOutputMaterial')
                bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')

                image_tex_node = nodes.new(type='ShaderNodeTexImage')
                image_tex_node.image = image_texture

                links.new(image_tex_node.outputs["Color"], bsdf_node.inputs["Base Color"])
                links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

                obj.data.materials.append(material)

                Backend.shared_close()
            
            def on_exception(_, exception):
                mesh_progress.active = False
                print(exception)
                raise exception

            future.add_response_callback(on_response)
            future.add_exception_callback(on_exception)
            future.add_done_callback(on_done)

            return {"FINISHED"}

    def register():
        bpy.utils.register_class(DreamModelProgress)
        bpy.utils.register_class(DreamModelSettings)
        bpy.types.Scene.dream_model_settings = bpy.props.PointerProperty(type=DreamModelSettings)
        
        bpy.utils.register_class(DreamModelPanel)
        
        bpy.utils.register_class(DreamModelCancel)
        bpy.utils.register_class(GeneratePointCloud)
        bpy.utils.register_class(GenerateMesh)

    def unregister():
        del bpy.types.Scene.dream_model_settings
        bpy.utils.unregister_class(DreamModelProgress)
        bpy.utils.unregister_class(DreamModelSettings)
        
        bpy.utils.unregister_class(DreamModelPanel)

        bpy.utils.unregister_class(DreamModelCancel)
        bpy.utils.unregister_class(GeneratePointCloud)
        bpy.utils.unregister_class(GenerateMesh)

        Backend.shared_close()