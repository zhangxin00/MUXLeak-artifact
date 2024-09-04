// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1ns/1ps
module fpga_top_axilite_s_axi
#(parameter
    C_S_AXI_ADDR_WIDTH = 8,
    C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire                          interrupt,
    output wire                          ap_start,
    input  wire                          ap_done,
    input  wire                          ap_ready,
    input  wire                          ap_idle,
    input  wire [2:0]                    layer_name_address0,
    input  wire                          layer_name_ce0,
    input  wire                          layer_name_we0,
    input  wire [7:0]                    layer_name_d0,
    output wire [8:0]                    layer_width_V,
    output wire [8:0]                    layer_height_V,
    output wire [9:0]                    layer_channels_in_V,
    output wire [9:0]                    layer_channels_out_V,
    output wire [1:0]                    layer_kernel_V,
    output wire [1:0]                    layer_stride_V,
    output wire [0:0]                    layer_pad,
    output wire [0:0]                    layer_relu,
    output wire [0:0]                    layer_is_first_split_layer,
    output wire [0:0]                    layer_is_second_split_layer,
    output wire [0:0]                    layer_global_pool,
    output wire [22:0]                   layer_mem_addr_input_V,
    output wire [22:0]                   layer_mem_addr_output_V,
    output wire [22:0]                   layer_mem_addr_weights_V,
    output wire [31:0]                   SHARED_DRAM,
    output wire [31:0]                   weights_offset,
    output wire [18:0]                   num_weights_V,
    output wire [31:0]                   input_offset
);
//------------------------Address Info-------------------
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x18 : Data signal of layer_width_V
//        bit 8~0 - layer_width_V[8:0] (Read/Write)
//        others  - reserved
// 0x1c : reserved
// 0x20 : Data signal of layer_height_V
//        bit 8~0 - layer_height_V[8:0] (Read/Write)
//        others  - reserved
// 0x24 : reserved
// 0x28 : Data signal of layer_channels_in_V
//        bit 9~0 - layer_channels_in_V[9:0] (Read/Write)
//        others  - reserved
// 0x2c : reserved
// 0x30 : Data signal of layer_channels_out_V
//        bit 9~0 - layer_channels_out_V[9:0] (Read/Write)
//        others  - reserved
// 0x34 : reserved
// 0x38 : Data signal of layer_kernel_V
//        bit 1~0 - layer_kernel_V[1:0] (Read/Write)
//        others  - reserved
// 0x3c : reserved
// 0x40 : Data signal of layer_stride_V
//        bit 1~0 - layer_stride_V[1:0] (Read/Write)
//        others  - reserved
// 0x44 : reserved
// 0x48 : Data signal of layer_pad
//        bit 0  - layer_pad[0] (Read/Write)
//        others - reserved
// 0x4c : reserved
// 0x50 : Data signal of layer_relu
//        bit 0  - layer_relu[0] (Read/Write)
//        others - reserved
// 0x54 : reserved
// 0x58 : Data signal of layer_is_first_split_layer
//        bit 0  - layer_is_first_split_layer[0] (Read/Write)
//        others - reserved
// 0x5c : reserved
// 0x60 : Data signal of layer_is_second_split_layer
//        bit 0  - layer_is_second_split_layer[0] (Read/Write)
//        others - reserved
// 0x64 : reserved
// 0x68 : Data signal of layer_global_pool
//        bit 0  - layer_global_pool[0] (Read/Write)
//        others - reserved
// 0x6c : reserved
// 0x70 : Data signal of layer_mem_addr_input_V
//        bit 22~0 - layer_mem_addr_input_V[22:0] (Read/Write)
//        others   - reserved
// 0x74 : reserved
// 0x78 : Data signal of layer_mem_addr_output_V
//        bit 22~0 - layer_mem_addr_output_V[22:0] (Read/Write)
//        others   - reserved
// 0x7c : reserved
// 0x80 : Data signal of layer_mem_addr_weights_V
//        bit 22~0 - layer_mem_addr_weights_V[22:0] (Read/Write)
//        others   - reserved
// 0x84 : reserved
// 0x88 : Data signal of SHARED_DRAM
//        bit 31~0 - SHARED_DRAM[31:0] (Read/Write)
// 0x8c : reserved
// 0x90 : Data signal of weights_offset
//        bit 31~0 - weights_offset[31:0] (Read/Write)
// 0x94 : reserved
// 0x98 : Data signal of num_weights_V
//        bit 18~0 - num_weights_V[18:0] (Read/Write)
//        others   - reserved
// 0x9c : reserved
// 0xa0 : Data signal of input_offset
//        bit 31~0 - input_offset[31:0] (Read/Write)
// 0xa4 : reserved
// 0x10 ~
// 0x17 : Memory 'layer_name' (7 * 8b)
//        Word n : bit [ 7: 0] - layer_name[4n]
//                 bit [15: 8] - layer_name[4n+1]
//                 bit [23:16] - layer_name[4n+2]
//                 bit [31:24] - layer_name[4n+3]
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_AP_CTRL                            = 8'h00,
    ADDR_GIE                                = 8'h04,
    ADDR_IER                                = 8'h08,
    ADDR_ISR                                = 8'h0c,
    ADDR_LAYER_WIDTH_V_DATA_0               = 8'h18,
    ADDR_LAYER_WIDTH_V_CTRL                 = 8'h1c,
    ADDR_LAYER_HEIGHT_V_DATA_0              = 8'h20,
    ADDR_LAYER_HEIGHT_V_CTRL                = 8'h24,
    ADDR_LAYER_CHANNELS_IN_V_DATA_0         = 8'h28,
    ADDR_LAYER_CHANNELS_IN_V_CTRL           = 8'h2c,
    ADDR_LAYER_CHANNELS_OUT_V_DATA_0        = 8'h30,
    ADDR_LAYER_CHANNELS_OUT_V_CTRL          = 8'h34,
    ADDR_LAYER_KERNEL_V_DATA_0              = 8'h38,
    ADDR_LAYER_KERNEL_V_CTRL                = 8'h3c,
    ADDR_LAYER_STRIDE_V_DATA_0              = 8'h40,
    ADDR_LAYER_STRIDE_V_CTRL                = 8'h44,
    ADDR_LAYER_PAD_DATA_0                   = 8'h48,
    ADDR_LAYER_PAD_CTRL                     = 8'h4c,
    ADDR_LAYER_RELU_DATA_0                  = 8'h50,
    ADDR_LAYER_RELU_CTRL                    = 8'h54,
    ADDR_LAYER_IS_FIRST_SPLIT_LAYER_DATA_0  = 8'h58,
    ADDR_LAYER_IS_FIRST_SPLIT_LAYER_CTRL    = 8'h5c,
    ADDR_LAYER_IS_SECOND_SPLIT_LAYER_DATA_0 = 8'h60,
    ADDR_LAYER_IS_SECOND_SPLIT_LAYER_CTRL   = 8'h64,
    ADDR_LAYER_GLOBAL_POOL_DATA_0           = 8'h68,
    ADDR_LAYER_GLOBAL_POOL_CTRL             = 8'h6c,
    ADDR_LAYER_MEM_ADDR_INPUT_V_DATA_0      = 8'h70,
    ADDR_LAYER_MEM_ADDR_INPUT_V_CTRL        = 8'h74,
    ADDR_LAYER_MEM_ADDR_OUTPUT_V_DATA_0     = 8'h78,
    ADDR_LAYER_MEM_ADDR_OUTPUT_V_CTRL       = 8'h7c,
    ADDR_LAYER_MEM_ADDR_WEIGHTS_V_DATA_0    = 8'h80,
    ADDR_LAYER_MEM_ADDR_WEIGHTS_V_CTRL      = 8'h84,
    ADDR_SHARED_DRAM_DATA_0                 = 8'h88,
    ADDR_SHARED_DRAM_CTRL                   = 8'h8c,
    ADDR_WEIGHTS_OFFSET_DATA_0              = 8'h90,
    ADDR_WEIGHTS_OFFSET_CTRL                = 8'h94,
    ADDR_NUM_WEIGHTS_V_DATA_0               = 8'h98,
    ADDR_NUM_WEIGHTS_V_CTRL                 = 8'h9c,
    ADDR_INPUT_OFFSET_DATA_0                = 8'ha0,
    ADDR_INPUT_OFFSET_CTRL                  = 8'ha4,
    ADDR_LAYER_NAME_BASE                    = 8'h10,
    ADDR_LAYER_NAME_HIGH                    = 8'h17,
    WRIDLE                                  = 2'd0,
    WRDATA                                  = 2'd1,
    WRRESP                                  = 2'd2,
    WRRESET                                 = 2'd3,
    RDIDLE                                  = 2'd0,
    RDDATA                                  = 2'd1,
    RDRESET                                 = 2'd2,
    ADDR_BITS         = 8;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRRESET;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [31:0]                   wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDRESET;
    reg  [1:0]                    rnext;
    reg  [31:0]                   rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    reg                           int_ap_idle;
    reg                           int_ap_ready;
    reg                           int_ap_done = 1'b0;
    reg                           int_ap_start = 1'b0;
    reg                           int_auto_restart = 1'b0;
    reg                           int_gie = 1'b0;
    reg  [1:0]                    int_ier = 2'b0;
    reg  [1:0]                    int_isr = 2'b0;
    reg  [8:0]                    int_layer_width_V = 'b0;
    reg  [8:0]                    int_layer_height_V = 'b0;
    reg  [9:0]                    int_layer_channels_in_V = 'b0;
    reg  [9:0]                    int_layer_channels_out_V = 'b0;
    reg  [1:0]                    int_layer_kernel_V = 'b0;
    reg  [1:0]                    int_layer_stride_V = 'b0;
    reg  [0:0]                    int_layer_pad = 'b0;
    reg  [0:0]                    int_layer_relu = 'b0;
    reg  [0:0]                    int_layer_is_first_split_layer = 'b0;
    reg  [0:0]                    int_layer_is_second_split_layer = 'b0;
    reg  [0:0]                    int_layer_global_pool = 'b0;
    reg  [22:0]                   int_layer_mem_addr_input_V = 'b0;
    reg  [22:0]                   int_layer_mem_addr_output_V = 'b0;
    reg  [22:0]                   int_layer_mem_addr_weights_V = 'b0;
    reg  [31:0]                   int_SHARED_DRAM = 'b0;
    reg  [31:0]                   int_weights_offset = 'b0;
    reg  [18:0]                   int_num_weights_V = 'b0;
    reg  [31:0]                   int_input_offset = 'b0;
    // memory signals
    wire [0:0]                    int_layer_name_address0;
    wire                          int_layer_name_ce0;
    wire                          int_layer_name_we0;
    wire [3:0]                    int_layer_name_be0;
    wire [31:0]                   int_layer_name_d0;
    wire [31:0]                   int_layer_name_q0;
    wire [0:0]                    int_layer_name_address1;
    wire                          int_layer_name_ce1;
    wire                          int_layer_name_we1;
    wire [3:0]                    int_layer_name_be1;
    wire [31:0]                   int_layer_name_d1;
    wire [31:0]                   int_layer_name_q1;
    reg                           int_layer_name_read;
    reg                           int_layer_name_write;
    reg  [1:0]                    int_layer_name_shift;

//------------------------Instantiation------------------
// int_layer_name
fpga_top_axilite_s_axi_ram #(
    .BYTES    ( 4 ),
    .DEPTH    ( 2 )
) int_layer_name (
    .clk0     ( ACLK ),
    .address0 ( int_layer_name_address0 ),
    .ce0      ( int_layer_name_ce0 ),
    .we0      ( int_layer_name_we0 ),
    .be0      ( int_layer_name_be0 ),
    .d0       ( int_layer_name_d0 ),
    .q0       ( int_layer_name_q0 ),
    .clk1     ( ACLK ),
    .address1 ( int_layer_name_address1 ),
    .ce1      ( int_layer_name_ce1 ),
    .we1      ( int_layer_name_we1 ),
    .be1      ( int_layer_name_be1 ),
    .d1       ( int_layer_name_d1 ),
    .q1       ( int_layer_name_q1 )
);

//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BRESP   = 2'b00;  // OKAY
assign BVALID  = (wstate == WRRESP);
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= AWADDR[ADDR_BITS-1:0];
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA) & !int_layer_name_read;
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 1'b0;
            case (raddr)
                ADDR_AP_CTRL: begin
                    rdata[0] <= int_ap_start;
                    rdata[1] <= int_ap_done;
                    rdata[2] <= int_ap_idle;
                    rdata[3] <= int_ap_ready;
                    rdata[7] <= int_auto_restart;
                end
                ADDR_GIE: begin
                    rdata <= int_gie;
                end
                ADDR_IER: begin
                    rdata <= int_ier;
                end
                ADDR_ISR: begin
                    rdata <= int_isr;
                end
                ADDR_LAYER_WIDTH_V_DATA_0: begin
                    rdata <= int_layer_width_V[8:0];
                end
                ADDR_LAYER_HEIGHT_V_DATA_0: begin
                    rdata <= int_layer_height_V[8:0];
                end
                ADDR_LAYER_CHANNELS_IN_V_DATA_0: begin
                    rdata <= int_layer_channels_in_V[9:0];
                end
                ADDR_LAYER_CHANNELS_OUT_V_DATA_0: begin
                    rdata <= int_layer_channels_out_V[9:0];
                end
                ADDR_LAYER_KERNEL_V_DATA_0: begin
                    rdata <= int_layer_kernel_V[1:0];
                end
                ADDR_LAYER_STRIDE_V_DATA_0: begin
                    rdata <= int_layer_stride_V[1:0];
                end
                ADDR_LAYER_PAD_DATA_0: begin
                    rdata <= int_layer_pad[0:0];
                end
                ADDR_LAYER_RELU_DATA_0: begin
                    rdata <= int_layer_relu[0:0];
                end
                ADDR_LAYER_IS_FIRST_SPLIT_LAYER_DATA_0: begin
                    rdata <= int_layer_is_first_split_layer[0:0];
                end
                ADDR_LAYER_IS_SECOND_SPLIT_LAYER_DATA_0: begin
                    rdata <= int_layer_is_second_split_layer[0:0];
                end
                ADDR_LAYER_GLOBAL_POOL_DATA_0: begin
                    rdata <= int_layer_global_pool[0:0];
                end
                ADDR_LAYER_MEM_ADDR_INPUT_V_DATA_0: begin
                    rdata <= int_layer_mem_addr_input_V[22:0];
                end
                ADDR_LAYER_MEM_ADDR_OUTPUT_V_DATA_0: begin
                    rdata <= int_layer_mem_addr_output_V[22:0];
                end
                ADDR_LAYER_MEM_ADDR_WEIGHTS_V_DATA_0: begin
                    rdata <= int_layer_mem_addr_weights_V[22:0];
                end
                ADDR_SHARED_DRAM_DATA_0: begin
                    rdata <= int_SHARED_DRAM[31:0];
                end
                ADDR_WEIGHTS_OFFSET_DATA_0: begin
                    rdata <= int_weights_offset[31:0];
                end
                ADDR_NUM_WEIGHTS_V_DATA_0: begin
                    rdata <= int_num_weights_V[18:0];
                end
                ADDR_INPUT_OFFSET_DATA_0: begin
                    rdata <= int_input_offset[31:0];
                end
            endcase
        end
        else if (int_layer_name_read) begin
            rdata <= int_layer_name_q1;
        end
    end
end


//------------------------Register logic-----------------
assign interrupt                   = int_gie & (|int_isr);
assign ap_start                    = int_ap_start;
assign layer_width_V               = int_layer_width_V;
assign layer_height_V              = int_layer_height_V;
assign layer_channels_in_V         = int_layer_channels_in_V;
assign layer_channels_out_V        = int_layer_channels_out_V;
assign layer_kernel_V              = int_layer_kernel_V;
assign layer_stride_V              = int_layer_stride_V;
assign layer_pad                   = int_layer_pad;
assign layer_relu                  = int_layer_relu;
assign layer_is_first_split_layer  = int_layer_is_first_split_layer;
assign layer_is_second_split_layer = int_layer_is_second_split_layer;
assign layer_global_pool           = int_layer_global_pool;
assign layer_mem_addr_input_V      = int_layer_mem_addr_input_V;
assign layer_mem_addr_output_V     = int_layer_mem_addr_output_V;
assign layer_mem_addr_weights_V    = int_layer_mem_addr_weights_V;
assign SHARED_DRAM                 = int_SHARED_DRAM;
assign weights_offset              = int_weights_offset;
assign num_weights_V               = int_num_weights_V;
assign input_offset                = int_input_offset;
// int_ap_start
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
        else if (ap_ready)
            int_ap_start <= int_auto_restart; // clear on handshake/auto restart
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
        if (ap_done)
            int_ap_done <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_ap_done <= 1'b0; // clear on read
    end
end

// int_ap_idle
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_idle <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_idle <= ap_idle;
    end
end

// int_ap_ready
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_ready <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_ready <= ap_ready;
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0])
            int_auto_restart <=  WDATA[7];
    end
end

// int_gie
always @(posedge ACLK) begin
    if (ARESET)
        int_gie <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GIE && WSTRB[0])
            int_gie <= WDATA[0];
    end
end

// int_ier
always @(posedge ACLK) begin
    if (ARESET)
        int_ier <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IER && WSTRB[0])
            int_ier <= WDATA[1:0];
    end
end

// int_isr[0]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[0] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[0] & ap_done)
            int_isr[0] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[0] <= int_isr[0] ^ WDATA[0]; // toggle on write
    end
end

// int_isr[1]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[1] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[1] & ap_ready)
            int_isr[1] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[1] <= int_isr[1] ^ WDATA[1]; // toggle on write
    end
end

// int_layer_width_V[8:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_width_V[8:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_WIDTH_V_DATA_0)
            int_layer_width_V[8:0] <= (WDATA[31:0] & wmask) | (int_layer_width_V[8:0] & ~wmask);
    end
end

// int_layer_height_V[8:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_height_V[8:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_HEIGHT_V_DATA_0)
            int_layer_height_V[8:0] <= (WDATA[31:0] & wmask) | (int_layer_height_V[8:0] & ~wmask);
    end
end

// int_layer_channels_in_V[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_channels_in_V[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_CHANNELS_IN_V_DATA_0)
            int_layer_channels_in_V[9:0] <= (WDATA[31:0] & wmask) | (int_layer_channels_in_V[9:0] & ~wmask);
    end
end

// int_layer_channels_out_V[9:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_channels_out_V[9:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_CHANNELS_OUT_V_DATA_0)
            int_layer_channels_out_V[9:0] <= (WDATA[31:0] & wmask) | (int_layer_channels_out_V[9:0] & ~wmask);
    end
end

// int_layer_kernel_V[1:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_kernel_V[1:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_KERNEL_V_DATA_0)
            int_layer_kernel_V[1:0] <= (WDATA[31:0] & wmask) | (int_layer_kernel_V[1:0] & ~wmask);
    end
end

// int_layer_stride_V[1:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_stride_V[1:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_STRIDE_V_DATA_0)
            int_layer_stride_V[1:0] <= (WDATA[31:0] & wmask) | (int_layer_stride_V[1:0] & ~wmask);
    end
end

// int_layer_pad[0:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_pad[0:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_PAD_DATA_0)
            int_layer_pad[0:0] <= (WDATA[31:0] & wmask) | (int_layer_pad[0:0] & ~wmask);
    end
end

// int_layer_relu[0:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_relu[0:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_RELU_DATA_0)
            int_layer_relu[0:0] <= (WDATA[31:0] & wmask) | (int_layer_relu[0:0] & ~wmask);
    end
end

// int_layer_is_first_split_layer[0:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_is_first_split_layer[0:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_IS_FIRST_SPLIT_LAYER_DATA_0)
            int_layer_is_first_split_layer[0:0] <= (WDATA[31:0] & wmask) | (int_layer_is_first_split_layer[0:0] & ~wmask);
    end
end

// int_layer_is_second_split_layer[0:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_is_second_split_layer[0:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_IS_SECOND_SPLIT_LAYER_DATA_0)
            int_layer_is_second_split_layer[0:0] <= (WDATA[31:0] & wmask) | (int_layer_is_second_split_layer[0:0] & ~wmask);
    end
end

// int_layer_global_pool[0:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_global_pool[0:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_GLOBAL_POOL_DATA_0)
            int_layer_global_pool[0:0] <= (WDATA[31:0] & wmask) | (int_layer_global_pool[0:0] & ~wmask);
    end
end

// int_layer_mem_addr_input_V[22:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_mem_addr_input_V[22:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_MEM_ADDR_INPUT_V_DATA_0)
            int_layer_mem_addr_input_V[22:0] <= (WDATA[31:0] & wmask) | (int_layer_mem_addr_input_V[22:0] & ~wmask);
    end
end

// int_layer_mem_addr_output_V[22:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_mem_addr_output_V[22:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_MEM_ADDR_OUTPUT_V_DATA_0)
            int_layer_mem_addr_output_V[22:0] <= (WDATA[31:0] & wmask) | (int_layer_mem_addr_output_V[22:0] & ~wmask);
    end
end

// int_layer_mem_addr_weights_V[22:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_mem_addr_weights_V[22:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LAYER_MEM_ADDR_WEIGHTS_V_DATA_0)
            int_layer_mem_addr_weights_V[22:0] <= (WDATA[31:0] & wmask) | (int_layer_mem_addr_weights_V[22:0] & ~wmask);
    end
end

// int_SHARED_DRAM[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_SHARED_DRAM[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SHARED_DRAM_DATA_0)
            int_SHARED_DRAM[31:0] <= (WDATA[31:0] & wmask) | (int_SHARED_DRAM[31:0] & ~wmask);
    end
end

// int_weights_offset[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_weights_offset[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_WEIGHTS_OFFSET_DATA_0)
            int_weights_offset[31:0] <= (WDATA[31:0] & wmask) | (int_weights_offset[31:0] & ~wmask);
    end
end

// int_num_weights_V[18:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_num_weights_V[18:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_NUM_WEIGHTS_V_DATA_0)
            int_num_weights_V[18:0] <= (WDATA[31:0] & wmask) | (int_num_weights_V[18:0] & ~wmask);
    end
end

// int_input_offset[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_input_offset[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_INPUT_OFFSET_DATA_0)
            int_input_offset[31:0] <= (WDATA[31:0] & wmask) | (int_input_offset[31:0] & ~wmask);
    end
end


//------------------------Memory logic-------------------
// layer_name
assign int_layer_name_address0 = layer_name_address0 >> 2;
assign int_layer_name_ce0      = layer_name_ce0;
assign int_layer_name_we0      = layer_name_we0;
assign int_layer_name_be0      = 1 << layer_name_address0[1:0];
assign int_layer_name_d0       = {4{layer_name_d0}};
assign int_layer_name_address1 = ar_hs? raddr[2:2] : waddr[2:2];
assign int_layer_name_ce1      = ar_hs | (int_layer_name_write & WVALID);
assign int_layer_name_we1      = int_layer_name_write & WVALID;
assign int_layer_name_be1      = WSTRB;
assign int_layer_name_d1       = WDATA;
// int_layer_name_read
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_name_read <= 1'b0;
    else if (ACLK_EN) begin
        if (ar_hs && raddr >= ADDR_LAYER_NAME_BASE && raddr <= ADDR_LAYER_NAME_HIGH)
            int_layer_name_read <= 1'b1;
        else
            int_layer_name_read <= 1'b0;
    end
end

// int_layer_name_write
always @(posedge ACLK) begin
    if (ARESET)
        int_layer_name_write <= 1'b0;
    else if (ACLK_EN) begin
        if (aw_hs && AWADDR[ADDR_BITS-1:0] >= ADDR_LAYER_NAME_BASE && AWADDR[ADDR_BITS-1:0] <= ADDR_LAYER_NAME_HIGH)
            int_layer_name_write <= 1'b1;
        else if (WVALID)
            int_layer_name_write <= 1'b0;
    end
end

// int_layer_name_shift
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (layer_name_ce0)
            int_layer_name_shift <= layer_name_address0[1:0];
    end
end


endmodule


`timescale 1ns/1ps

module fpga_top_axilite_s_axi_ram
#(parameter
    BYTES  = 4,
    DEPTH  = 256,
    AWIDTH = log2(DEPTH)
) (
    input  wire               clk0,
    input  wire [AWIDTH-1:0]  address0,
    input  wire               ce0,
    input  wire               we0,
    input  wire [BYTES-1:0]   be0,
    input  wire [BYTES*8-1:0] d0,
    output reg  [BYTES*8-1:0] q0,
    input  wire               clk1,
    input  wire [AWIDTH-1:0]  address1,
    input  wire               ce1,
    input  wire               we1,
    input  wire [BYTES-1:0]   be1,
    input  wire [BYTES*8-1:0] d1,
    output reg  [BYTES*8-1:0] q1
);
//------------------------Local signal-------------------
reg  [BYTES*8-1:0] mem[0:DEPTH-1];
//------------------------Task and function--------------
function integer log2;
    input integer x;
    integer n, m;
begin
    n = 1;
    m = 2;
    while (m < x) begin
        n = n + 1;
        m = m * 2;
    end
    log2 = n;
end
endfunction
//------------------------Body---------------------------
// read port 0
always @(posedge clk0) begin
    if (ce0) q0 <= mem[address0];
end

// read port 1
always @(posedge clk1) begin
    if (ce1) q1 <= mem[address1];
end

genvar i;
generate
    for (i = 0; i < BYTES; i = i + 1) begin : gen_write
        // write port 0
        always @(posedge clk0) begin
            if (ce0 & we0 & be0[i]) begin
                mem[address0][8*i+7:8*i] <= d0[8*i+7:8*i];
            end
        end
        // write port 1
        always @(posedge clk1) begin
            if (ce1 & we1 & be1[i]) begin
                mem[address1][8*i+7:8*i] <= d1[8*i+7:8*i];
            end
        end
    end
endgenerate

endmodule

