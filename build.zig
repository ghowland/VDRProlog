const std = @import("std");

pub fn build(b: *std.Build) void {
    const ptx_kernel = b.addObject(.{
        .name = "vlp_kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vlp_kernel.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .freestanding,
                .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_75 },
            }),
            .optimize = .Debug,
        }),
    });

    const install_ptx = b.addInstallFile(ptx_kernel.getEmittedAsm(), "vlp_kernel.ptx");
    b.getInstallStep().dependOn(&install_ptx.step);
}
