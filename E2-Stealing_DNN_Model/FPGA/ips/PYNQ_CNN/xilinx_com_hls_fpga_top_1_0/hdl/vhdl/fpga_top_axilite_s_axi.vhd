-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
-- Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity fpga_top_axilite_s_axi is
generic (
    C_S_AXI_ADDR_WIDTH    : INTEGER := 8;
    C_S_AXI_DATA_WIDTH    : INTEGER := 32);
port (
    ACLK                  :in   STD_LOGIC;
    ARESET                :in   STD_LOGIC;
    ACLK_EN               :in   STD_LOGIC;
    AWADDR                :in   STD_LOGIC_VECTOR(C_S_AXI_ADDR_WIDTH-1 downto 0);
    AWVALID               :in   STD_LOGIC;
    AWREADY               :out  STD_LOGIC;
    WDATA                 :in   STD_LOGIC_VECTOR(C_S_AXI_DATA_WIDTH-1 downto 0);
    WSTRB                 :in   STD_LOGIC_VECTOR(C_S_AXI_DATA_WIDTH/8-1 downto 0);
    WVALID                :in   STD_LOGIC;
    WREADY                :out  STD_LOGIC;
    BRESP                 :out  STD_LOGIC_VECTOR(1 downto 0);
    BVALID                :out  STD_LOGIC;
    BREADY                :in   STD_LOGIC;
    ARADDR                :in   STD_LOGIC_VECTOR(C_S_AXI_ADDR_WIDTH-1 downto 0);
    ARVALID               :in   STD_LOGIC;
    ARREADY               :out  STD_LOGIC;
    RDATA                 :out  STD_LOGIC_VECTOR(C_S_AXI_DATA_WIDTH-1 downto 0);
    RRESP                 :out  STD_LOGIC_VECTOR(1 downto 0);
    RVALID                :out  STD_LOGIC;
    RREADY                :in   STD_LOGIC;
    interrupt             :out  STD_LOGIC;
    ap_start              :out  STD_LOGIC;
    ap_done               :in   STD_LOGIC;
    ap_ready              :in   STD_LOGIC;
    ap_idle               :in   STD_LOGIC;
    layer_name_address0   :in   STD_LOGIC_VECTOR(2 downto 0);
    layer_name_ce0        :in   STD_LOGIC;
    layer_name_we0        :in   STD_LOGIC;
    layer_name_d0         :in   STD_LOGIC_VECTOR(7 downto 0);
    layer_width_V         :out  STD_LOGIC_VECTOR(8 downto 0);
    layer_height_V        :out  STD_LOGIC_VECTOR(8 downto 0);
    layer_channels_in_V   :out  STD_LOGIC_VECTOR(9 downto 0);
    layer_channels_out_V  :out  STD_LOGIC_VECTOR(9 downto 0);
    layer_kernel_V        :out  STD_LOGIC_VECTOR(1 downto 0);
    layer_stride_V        :out  STD_LOGIC_VECTOR(1 downto 0);
    layer_pad             :out  STD_LOGIC_VECTOR(0 downto 0);
    layer_relu            :out  STD_LOGIC_VECTOR(0 downto 0);
    layer_is_first_split_layer :out  STD_LOGIC_VECTOR(0 downto 0);
    layer_is_second_split_layer :out  STD_LOGIC_VECTOR(0 downto 0);
    layer_global_pool     :out  STD_LOGIC_VECTOR(0 downto 0);
    layer_mem_addr_input_V :out  STD_LOGIC_VECTOR(22 downto 0);
    layer_mem_addr_output_V :out  STD_LOGIC_VECTOR(22 downto 0);
    layer_mem_addr_weights_V :out  STD_LOGIC_VECTOR(22 downto 0);
    SHARED_DRAM           :out  STD_LOGIC_VECTOR(31 downto 0);
    weights_offset        :out  STD_LOGIC_VECTOR(31 downto 0);
    num_weights_V         :out  STD_LOGIC_VECTOR(18 downto 0);
    input_offset          :out  STD_LOGIC_VECTOR(31 downto 0)
);
end entity fpga_top_axilite_s_axi;

-- ------------------------Address Info-------------------
-- 0x00 : Control signals
--        bit 0  - ap_start (Read/Write/COH)
--        bit 1  - ap_done (Read/COR)
--        bit 2  - ap_idle (Read)
--        bit 3  - ap_ready (Read)
--        bit 7  - auto_restart (Read/Write)
--        others - reserved
-- 0x04 : Global Interrupt Enable Register
--        bit 0  - Global Interrupt Enable (Read/Write)
--        others - reserved
-- 0x08 : IP Interrupt Enable Register (Read/Write)
--        bit 0  - Channel 0 (ap_done)
--        bit 1  - Channel 1 (ap_ready)
--        others - reserved
-- 0x0c : IP Interrupt Status Register (Read/TOW)
--        bit 0  - Channel 0 (ap_done)
--        bit 1  - Channel 1 (ap_ready)
--        others - reserved
-- 0x18 : Data signal of layer_width_V
--        bit 8~0 - layer_width_V[8:0] (Read/Write)
--        others  - reserved
-- 0x1c : reserved
-- 0x20 : Data signal of layer_height_V
--        bit 8~0 - layer_height_V[8:0] (Read/Write)
--        others  - reserved
-- 0x24 : reserved
-- 0x28 : Data signal of layer_channels_in_V
--        bit 9~0 - layer_channels_in_V[9:0] (Read/Write)
--        others  - reserved
-- 0x2c : reserved
-- 0x30 : Data signal of layer_channels_out_V
--        bit 9~0 - layer_channels_out_V[9:0] (Read/Write)
--        others  - reserved
-- 0x34 : reserved
-- 0x38 : Data signal of layer_kernel_V
--        bit 1~0 - layer_kernel_V[1:0] (Read/Write)
--        others  - reserved
-- 0x3c : reserved
-- 0x40 : Data signal of layer_stride_V
--        bit 1~0 - layer_stride_V[1:0] (Read/Write)
--        others  - reserved
-- 0x44 : reserved
-- 0x48 : Data signal of layer_pad
--        bit 0  - layer_pad[0] (Read/Write)
--        others - reserved
-- 0x4c : reserved
-- 0x50 : Data signal of layer_relu
--        bit 0  - layer_relu[0] (Read/Write)
--        others - reserved
-- 0x54 : reserved
-- 0x58 : Data signal of layer_is_first_split_layer
--        bit 0  - layer_is_first_split_layer[0] (Read/Write)
--        others - reserved
-- 0x5c : reserved
-- 0x60 : Data signal of layer_is_second_split_layer
--        bit 0  - layer_is_second_split_layer[0] (Read/Write)
--        others - reserved
-- 0x64 : reserved
-- 0x68 : Data signal of layer_global_pool
--        bit 0  - layer_global_pool[0] (Read/Write)
--        others - reserved
-- 0x6c : reserved
-- 0x70 : Data signal of layer_mem_addr_input_V
--        bit 22~0 - layer_mem_addr_input_V[22:0] (Read/Write)
--        others   - reserved
-- 0x74 : reserved
-- 0x78 : Data signal of layer_mem_addr_output_V
--        bit 22~0 - layer_mem_addr_output_V[22:0] (Read/Write)
--        others   - reserved
-- 0x7c : reserved
-- 0x80 : Data signal of layer_mem_addr_weights_V
--        bit 22~0 - layer_mem_addr_weights_V[22:0] (Read/Write)
--        others   - reserved
-- 0x84 : reserved
-- 0x88 : Data signal of SHARED_DRAM
--        bit 31~0 - SHARED_DRAM[31:0] (Read/Write)
-- 0x8c : reserved
-- 0x90 : Data signal of weights_offset
--        bit 31~0 - weights_offset[31:0] (Read/Write)
-- 0x94 : reserved
-- 0x98 : Data signal of num_weights_V
--        bit 18~0 - num_weights_V[18:0] (Read/Write)
--        others   - reserved
-- 0x9c : reserved
-- 0xa0 : Data signal of input_offset
--        bit 31~0 - input_offset[31:0] (Read/Write)
-- 0xa4 : reserved
-- 0x10 ~
-- 0x17 : Memory 'layer_name' (7 * 8b)
--        Word n : bit [ 7: 0] - layer_name[4n]
--                 bit [15: 8] - layer_name[4n+1]
--                 bit [23:16] - layer_name[4n+2]
--                 bit [31:24] - layer_name[4n+3]
-- (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

architecture behave of fpga_top_axilite_s_axi is
    type states is (wridle, wrdata, wrresp, wrreset, rdidle, rddata, rdreset);  -- read and write fsm states
    signal wstate  : states := wrreset;
    signal rstate  : states := rdreset;
    signal wnext, rnext: states;
    constant ADDR_AP_CTRL                            : INTEGER := 16#00#;
    constant ADDR_GIE                                : INTEGER := 16#04#;
    constant ADDR_IER                                : INTEGER := 16#08#;
    constant ADDR_ISR                                : INTEGER := 16#0c#;
    constant ADDR_LAYER_WIDTH_V_DATA_0               : INTEGER := 16#18#;
    constant ADDR_LAYER_WIDTH_V_CTRL                 : INTEGER := 16#1c#;
    constant ADDR_LAYER_HEIGHT_V_DATA_0              : INTEGER := 16#20#;
    constant ADDR_LAYER_HEIGHT_V_CTRL                : INTEGER := 16#24#;
    constant ADDR_LAYER_CHANNELS_IN_V_DATA_0         : INTEGER := 16#28#;
    constant ADDR_LAYER_CHANNELS_IN_V_CTRL           : INTEGER := 16#2c#;
    constant ADDR_LAYER_CHANNELS_OUT_V_DATA_0        : INTEGER := 16#30#;
    constant ADDR_LAYER_CHANNELS_OUT_V_CTRL          : INTEGER := 16#34#;
    constant ADDR_LAYER_KERNEL_V_DATA_0              : INTEGER := 16#38#;
    constant ADDR_LAYER_KERNEL_V_CTRL                : INTEGER := 16#3c#;
    constant ADDR_LAYER_STRIDE_V_DATA_0              : INTEGER := 16#40#;
    constant ADDR_LAYER_STRIDE_V_CTRL                : INTEGER := 16#44#;
    constant ADDR_LAYER_PAD_DATA_0                   : INTEGER := 16#48#;
    constant ADDR_LAYER_PAD_CTRL                     : INTEGER := 16#4c#;
    constant ADDR_LAYER_RELU_DATA_0                  : INTEGER := 16#50#;
    constant ADDR_LAYER_RELU_CTRL                    : INTEGER := 16#54#;
    constant ADDR_LAYER_IS_FIRST_SPLIT_LAYER_DATA_0  : INTEGER := 16#58#;
    constant ADDR_LAYER_IS_FIRST_SPLIT_LAYER_CTRL    : INTEGER := 16#5c#;
    constant ADDR_LAYER_IS_SECOND_SPLIT_LAYER_DATA_0 : INTEGER := 16#60#;
    constant ADDR_LAYER_IS_SECOND_SPLIT_LAYER_CTRL   : INTEGER := 16#64#;
    constant ADDR_LAYER_GLOBAL_POOL_DATA_0           : INTEGER := 16#68#;
    constant ADDR_LAYER_GLOBAL_POOL_CTRL             : INTEGER := 16#6c#;
    constant ADDR_LAYER_MEM_ADDR_INPUT_V_DATA_0      : INTEGER := 16#70#;
    constant ADDR_LAYER_MEM_ADDR_INPUT_V_CTRL        : INTEGER := 16#74#;
    constant ADDR_LAYER_MEM_ADDR_OUTPUT_V_DATA_0     : INTEGER := 16#78#;
    constant ADDR_LAYER_MEM_ADDR_OUTPUT_V_CTRL       : INTEGER := 16#7c#;
    constant ADDR_LAYER_MEM_ADDR_WEIGHTS_V_DATA_0    : INTEGER := 16#80#;
    constant ADDR_LAYER_MEM_ADDR_WEIGHTS_V_CTRL      : INTEGER := 16#84#;
    constant ADDR_SHARED_DRAM_DATA_0                 : INTEGER := 16#88#;
    constant ADDR_SHARED_DRAM_CTRL                   : INTEGER := 16#8c#;
    constant ADDR_WEIGHTS_OFFSET_DATA_0              : INTEGER := 16#90#;
    constant ADDR_WEIGHTS_OFFSET_CTRL                : INTEGER := 16#94#;
    constant ADDR_NUM_WEIGHTS_V_DATA_0               : INTEGER := 16#98#;
    constant ADDR_NUM_WEIGHTS_V_CTRL                 : INTEGER := 16#9c#;
    constant ADDR_INPUT_OFFSET_DATA_0                : INTEGER := 16#a0#;
    constant ADDR_INPUT_OFFSET_CTRL                  : INTEGER := 16#a4#;
    constant ADDR_LAYER_NAME_BASE                    : INTEGER := 16#10#;
    constant ADDR_LAYER_NAME_HIGH                    : INTEGER := 16#17#;
    constant ADDR_BITS         : INTEGER := 8;

    signal waddr               : UNSIGNED(ADDR_BITS-1 downto 0);
    signal wmask               : UNSIGNED(31 downto 0);
    signal aw_hs               : STD_LOGIC;
    signal w_hs                : STD_LOGIC;
    signal rdata_data          : UNSIGNED(31 downto 0);
    signal ar_hs               : STD_LOGIC;
    signal raddr               : UNSIGNED(ADDR_BITS-1 downto 0);
    signal AWREADY_t           : STD_LOGIC;
    signal WREADY_t            : STD_LOGIC;
    signal ARREADY_t           : STD_LOGIC;
    signal RVALID_t            : STD_LOGIC;
    -- internal registers
    signal int_ap_idle         : STD_LOGIC;
    signal int_ap_ready        : STD_LOGIC;
    signal int_ap_done         : STD_LOGIC := '0';
    signal int_ap_start        : STD_LOGIC := '0';
    signal int_auto_restart    : STD_LOGIC := '0';
    signal int_gie             : STD_LOGIC := '0';
    signal int_ier             : UNSIGNED(1 downto 0) := (others => '0');
    signal int_isr             : UNSIGNED(1 downto 0) := (others => '0');
    signal int_layer_width_V   : UNSIGNED(8 downto 0) := (others => '0');
    signal int_layer_height_V  : UNSIGNED(8 downto 0) := (others => '0');
    signal int_layer_channels_in_V : UNSIGNED(9 downto 0) := (others => '0');
    signal int_layer_channels_out_V : UNSIGNED(9 downto 0) := (others => '0');
    signal int_layer_kernel_V  : UNSIGNED(1 downto 0) := (others => '0');
    signal int_layer_stride_V  : UNSIGNED(1 downto 0) := (others => '0');
    signal int_layer_pad       : UNSIGNED(0 downto 0) := (others => '0');
    signal int_layer_relu      : UNSIGNED(0 downto 0) := (others => '0');
    signal int_layer_is_first_split_layer : UNSIGNED(0 downto 0) := (others => '0');
    signal int_layer_is_second_split_layer : UNSIGNED(0 downto 0) := (others => '0');
    signal int_layer_global_pool : UNSIGNED(0 downto 0) := (others => '0');
    signal int_layer_mem_addr_input_V : UNSIGNED(22 downto 0) := (others => '0');
    signal int_layer_mem_addr_output_V : UNSIGNED(22 downto 0) := (others => '0');
    signal int_layer_mem_addr_weights_V : UNSIGNED(22 downto 0) := (others => '0');
    signal int_SHARED_DRAM     : UNSIGNED(31 downto 0) := (others => '0');
    signal int_weights_offset  : UNSIGNED(31 downto 0) := (others => '0');
    signal int_num_weights_V   : UNSIGNED(18 downto 0) := (others => '0');
    signal int_input_offset    : UNSIGNED(31 downto 0) := (others => '0');
    -- memory signals
    signal int_layer_name_address0 : UNSIGNED(0 downto 0);
    signal int_layer_name_ce0  : STD_LOGIC;
    signal int_layer_name_we0  : STD_LOGIC;
    signal int_layer_name_be0  : UNSIGNED(3 downto 0);
    signal int_layer_name_d0   : UNSIGNED(31 downto 0);
    signal int_layer_name_q0   : UNSIGNED(31 downto 0);
    signal int_layer_name_address1 : UNSIGNED(0 downto 0);
    signal int_layer_name_ce1  : STD_LOGIC;
    signal int_layer_name_we1  : STD_LOGIC;
    signal int_layer_name_be1  : UNSIGNED(3 downto 0);
    signal int_layer_name_d1   : UNSIGNED(31 downto 0);
    signal int_layer_name_q1   : UNSIGNED(31 downto 0);
    signal int_layer_name_read : STD_LOGIC;
    signal int_layer_name_write : STD_LOGIC;
    signal int_layer_name_shift : UNSIGNED(1 downto 0);

    component fpga_top_axilite_s_axi_ram is
        generic (
            BYTES   : INTEGER :=4;
            DEPTH   : INTEGER :=256;
            AWIDTH  : INTEGER :=8);
        port (
            clk0    : in  STD_LOGIC;
            address0: in  UNSIGNED(AWIDTH-1 downto 0);
            ce0     : in  STD_LOGIC;
            we0     : in  STD_LOGIC;
            be0     : in  UNSIGNED(BYTES-1 downto 0);
            d0      : in  UNSIGNED(BYTES*8-1 downto 0);
            q0      : out UNSIGNED(BYTES*8-1 downto 0);
            clk1    : in  STD_LOGIC;
            address1: in  UNSIGNED(AWIDTH-1 downto 0);
            ce1     : in  STD_LOGIC;
            we1     : in  STD_LOGIC;
            be1     : in  UNSIGNED(BYTES-1 downto 0);
            d1      : in  UNSIGNED(BYTES*8-1 downto 0);
            q1      : out UNSIGNED(BYTES*8-1 downto 0));
    end component fpga_top_axilite_s_axi_ram;

    function log2 (x : INTEGER) return INTEGER is
        variable n, m : INTEGER;
    begin
        n := 1;
        m := 2;
        while m < x loop
            n := n + 1;
            m := m * 2;
        end loop;
        return n;
    end function log2;

begin
-- ----------------------- Instantiation------------------
-- int_layer_name
int_layer_name : fpga_top_axilite_s_axi_ram
generic map (
     BYTES    => 4,
     DEPTH    => 2,
     AWIDTH   => log2(2))
port map (
     clk0     => ACLK,
     address0 => int_layer_name_address0,
     ce0      => int_layer_name_ce0,
     we0      => int_layer_name_we0,
     be0      => int_layer_name_be0,
     d0       => int_layer_name_d0,
     q0       => int_layer_name_q0,
     clk1     => ACLK,
     address1 => int_layer_name_address1,
     ce1      => int_layer_name_ce1,
     we1      => int_layer_name_we1,
     be1      => int_layer_name_be1,
     d1       => int_layer_name_d1,
     q1       => int_layer_name_q1);

-- ----------------------- AXI WRITE ---------------------
    AWREADY_t <=  '1' when wstate = wridle else '0';
    AWREADY   <=  AWREADY_t;
    WREADY_t  <=  '1' when wstate = wrdata else '0';
    WREADY    <=  WREADY_t;
    BRESP     <=  "00";  -- OKAY
    BVALID    <=  '1' when wstate = wrresp else '0';
    wmask     <=  (31 downto 24 => WSTRB(3), 23 downto 16 => WSTRB(2), 15 downto 8 => WSTRB(1), 7 downto 0 => WSTRB(0));
    aw_hs     <=  AWVALID and AWREADY_t;
    w_hs      <=  WVALID and WREADY_t;

    -- write FSM
    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                wstate <= wrreset;
            elsif (ACLK_EN = '1') then
                wstate <= wnext;
            end if;
        end if;
    end process;

    process (wstate, AWVALID, WVALID, BREADY)
    begin
        case (wstate) is
        when wridle =>
            if (AWVALID = '1') then
                wnext <= wrdata;
            else
                wnext <= wridle;
            end if;
        when wrdata =>
            if (WVALID = '1') then
                wnext <= wrresp;
            else
                wnext <= wrdata;
            end if;
        when wrresp =>
            if (BREADY = '1') then
                wnext <= wridle;
            else
                wnext <= wrresp;
            end if;
        when others =>
            wnext <= wridle;
        end case;
    end process;

    waddr_proc : process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (aw_hs = '1') then
                    waddr <= UNSIGNED(AWADDR(ADDR_BITS-1 downto 0));
                end if;
            end if;
        end if;
    end process;

-- ----------------------- AXI READ ----------------------
    ARREADY_t <= '1' when (rstate = rdidle) else '0';
    ARREADY <= ARREADY_t;
    RDATA   <= STD_LOGIC_VECTOR(rdata_data);
    RRESP   <= "00";  -- OKAY
    RVALID_t  <= '1' when (rstate = rddata) and (int_layer_name_read = '0') else '0';
    RVALID    <= RVALID_t;
    ar_hs   <= ARVALID and ARREADY_t;
    raddr   <= UNSIGNED(ARADDR(ADDR_BITS-1 downto 0));

    -- read FSM
    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                rstate <= rdreset;
            elsif (ACLK_EN = '1') then
                rstate <= rnext;
            end if;
        end if;
    end process;

    process (rstate, ARVALID, RREADY, RVALID_t)
    begin
        case (rstate) is
        when rdidle =>
            if (ARVALID = '1') then
                rnext <= rddata;
            else
                rnext <= rdidle;
            end if;
        when rddata =>
            if (RREADY = '1' and RVALID_t = '1') then
                rnext <= rdidle;
            else
                rnext <= rddata;
            end if;
        when others =>
            rnext <= rdidle;
        end case;
    end process;

    rdata_proc : process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (ar_hs = '1') then
                    case (TO_INTEGER(raddr)) is
                    when ADDR_AP_CTRL =>
                        rdata_data <= (7 => int_auto_restart, 3 => int_ap_ready, 2 => int_ap_idle, 1 => int_ap_done, 0 => int_ap_start, others => '0');
                    when ADDR_GIE =>
                        rdata_data <= (0 => int_gie, others => '0');
                    when ADDR_IER =>
                        rdata_data <= (1 => int_ier(1), 0 => int_ier(0), others => '0');
                    when ADDR_ISR =>
                        rdata_data <= (1 => int_isr(1), 0 => int_isr(0), others => '0');
                    when ADDR_LAYER_WIDTH_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_width_V(8 downto 0), 32);
                    when ADDR_LAYER_HEIGHT_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_height_V(8 downto 0), 32);
                    when ADDR_LAYER_CHANNELS_IN_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_channels_in_V(9 downto 0), 32);
                    when ADDR_LAYER_CHANNELS_OUT_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_channels_out_V(9 downto 0), 32);
                    when ADDR_LAYER_KERNEL_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_kernel_V(1 downto 0), 32);
                    when ADDR_LAYER_STRIDE_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_stride_V(1 downto 0), 32);
                    when ADDR_LAYER_PAD_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_pad(0 downto 0), 32);
                    when ADDR_LAYER_RELU_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_relu(0 downto 0), 32);
                    when ADDR_LAYER_IS_FIRST_SPLIT_LAYER_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_is_first_split_layer(0 downto 0), 32);
                    when ADDR_LAYER_IS_SECOND_SPLIT_LAYER_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_is_second_split_layer(0 downto 0), 32);
                    when ADDR_LAYER_GLOBAL_POOL_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_global_pool(0 downto 0), 32);
                    when ADDR_LAYER_MEM_ADDR_INPUT_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_mem_addr_input_V(22 downto 0), 32);
                    when ADDR_LAYER_MEM_ADDR_OUTPUT_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_mem_addr_output_V(22 downto 0), 32);
                    when ADDR_LAYER_MEM_ADDR_WEIGHTS_V_DATA_0 =>
                        rdata_data <= RESIZE(int_layer_mem_addr_weights_V(22 downto 0), 32);
                    when ADDR_SHARED_DRAM_DATA_0 =>
                        rdata_data <= RESIZE(int_SHARED_DRAM(31 downto 0), 32);
                    when ADDR_WEIGHTS_OFFSET_DATA_0 =>
                        rdata_data <= RESIZE(int_weights_offset(31 downto 0), 32);
                    when ADDR_NUM_WEIGHTS_V_DATA_0 =>
                        rdata_data <= RESIZE(int_num_weights_V(18 downto 0), 32);
                    when ADDR_INPUT_OFFSET_DATA_0 =>
                        rdata_data <= RESIZE(int_input_offset(31 downto 0), 32);
                    when others =>
                        rdata_data <= (others => '0');
                    end case;
                elsif (int_layer_name_read = '1') then
                    rdata_data <= int_layer_name_q1;
                end if;
            end if;
        end if;
    end process;

-- ----------------------- Register logic ----------------
    interrupt            <= int_gie and (int_isr(0) or int_isr(1));
    ap_start             <= int_ap_start;
    layer_width_V        <= STD_LOGIC_VECTOR(int_layer_width_V);
    layer_height_V       <= STD_LOGIC_VECTOR(int_layer_height_V);
    layer_channels_in_V  <= STD_LOGIC_VECTOR(int_layer_channels_in_V);
    layer_channels_out_V <= STD_LOGIC_VECTOR(int_layer_channels_out_V);
    layer_kernel_V       <= STD_LOGIC_VECTOR(int_layer_kernel_V);
    layer_stride_V       <= STD_LOGIC_VECTOR(int_layer_stride_V);
    layer_pad            <= STD_LOGIC_VECTOR(int_layer_pad);
    layer_relu           <= STD_LOGIC_VECTOR(int_layer_relu);
    layer_is_first_split_layer <= STD_LOGIC_VECTOR(int_layer_is_first_split_layer);
    layer_is_second_split_layer <= STD_LOGIC_VECTOR(int_layer_is_second_split_layer);
    layer_global_pool    <= STD_LOGIC_VECTOR(int_layer_global_pool);
    layer_mem_addr_input_V <= STD_LOGIC_VECTOR(int_layer_mem_addr_input_V);
    layer_mem_addr_output_V <= STD_LOGIC_VECTOR(int_layer_mem_addr_output_V);
    layer_mem_addr_weights_V <= STD_LOGIC_VECTOR(int_layer_mem_addr_weights_V);
    SHARED_DRAM          <= STD_LOGIC_VECTOR(int_SHARED_DRAM);
    weights_offset       <= STD_LOGIC_VECTOR(int_weights_offset);
    num_weights_V        <= STD_LOGIC_VECTOR(int_num_weights_V);
    input_offset         <= STD_LOGIC_VECTOR(int_input_offset);

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_ap_start <= '0';
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_AP_CTRL and WSTRB(0) = '1' and WDATA(0) = '1') then
                    int_ap_start <= '1';
                elsif (ap_ready = '1') then
                    int_ap_start <= int_auto_restart; -- clear on handshake/auto restart
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_ap_done <= '0';
            elsif (ACLK_EN = '1') then
                if (ap_done = '1') then
                    int_ap_done <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_AP_CTRL) then
                    int_ap_done <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_ap_idle <= '0';
            elsif (ACLK_EN = '1') then
                if (true) then
                    int_ap_idle <= ap_idle;
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_ap_ready <= '0';
            elsif (ACLK_EN = '1') then
                if (true) then
                    int_ap_ready <= ap_ready;
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_auto_restart <= '0';
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_AP_CTRL and WSTRB(0) = '1') then
                    int_auto_restart <= WDATA(7);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_gie <= '0';
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_GIE and WSTRB(0) = '1') then
                    int_gie <= WDATA(0);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_ier <= "00";
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_IER and WSTRB(0) = '1') then
                    int_ier <= UNSIGNED(WDATA(1 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_isr(0) <= '0';
            elsif (ACLK_EN = '1') then
                if (int_ier(0) = '1' and ap_done = '1') then
                    int_isr(0) <= '1';
                elsif (w_hs = '1' and waddr = ADDR_ISR and WSTRB(0) = '1') then
                    int_isr(0) <= int_isr(0) xor WDATA(0); -- toggle on write
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_isr(1) <= '0';
            elsif (ACLK_EN = '1') then
                if (int_ier(1) = '1' and ap_ready = '1') then
                    int_isr(1) <= '1';
                elsif (w_hs = '1' and waddr = ADDR_ISR and WSTRB(0) = '1') then
                    int_isr(1) <= int_isr(1) xor WDATA(1); -- toggle on write
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_WIDTH_V_DATA_0) then
                    int_layer_width_V(8 downto 0) <= (UNSIGNED(WDATA(8 downto 0)) and wmask(8 downto 0)) or ((not wmask(8 downto 0)) and int_layer_width_V(8 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_HEIGHT_V_DATA_0) then
                    int_layer_height_V(8 downto 0) <= (UNSIGNED(WDATA(8 downto 0)) and wmask(8 downto 0)) or ((not wmask(8 downto 0)) and int_layer_height_V(8 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_CHANNELS_IN_V_DATA_0) then
                    int_layer_channels_in_V(9 downto 0) <= (UNSIGNED(WDATA(9 downto 0)) and wmask(9 downto 0)) or ((not wmask(9 downto 0)) and int_layer_channels_in_V(9 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_CHANNELS_OUT_V_DATA_0) then
                    int_layer_channels_out_V(9 downto 0) <= (UNSIGNED(WDATA(9 downto 0)) and wmask(9 downto 0)) or ((not wmask(9 downto 0)) and int_layer_channels_out_V(9 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_KERNEL_V_DATA_0) then
                    int_layer_kernel_V(1 downto 0) <= (UNSIGNED(WDATA(1 downto 0)) and wmask(1 downto 0)) or ((not wmask(1 downto 0)) and int_layer_kernel_V(1 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_STRIDE_V_DATA_0) then
                    int_layer_stride_V(1 downto 0) <= (UNSIGNED(WDATA(1 downto 0)) and wmask(1 downto 0)) or ((not wmask(1 downto 0)) and int_layer_stride_V(1 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_PAD_DATA_0) then
                    int_layer_pad(0 downto 0) <= (UNSIGNED(WDATA(0 downto 0)) and wmask(0 downto 0)) or ((not wmask(0 downto 0)) and int_layer_pad(0 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_RELU_DATA_0) then
                    int_layer_relu(0 downto 0) <= (UNSIGNED(WDATA(0 downto 0)) and wmask(0 downto 0)) or ((not wmask(0 downto 0)) and int_layer_relu(0 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_IS_FIRST_SPLIT_LAYER_DATA_0) then
                    int_layer_is_first_split_layer(0 downto 0) <= (UNSIGNED(WDATA(0 downto 0)) and wmask(0 downto 0)) or ((not wmask(0 downto 0)) and int_layer_is_first_split_layer(0 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_IS_SECOND_SPLIT_LAYER_DATA_0) then
                    int_layer_is_second_split_layer(0 downto 0) <= (UNSIGNED(WDATA(0 downto 0)) and wmask(0 downto 0)) or ((not wmask(0 downto 0)) and int_layer_is_second_split_layer(0 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_GLOBAL_POOL_DATA_0) then
                    int_layer_global_pool(0 downto 0) <= (UNSIGNED(WDATA(0 downto 0)) and wmask(0 downto 0)) or ((not wmask(0 downto 0)) and int_layer_global_pool(0 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_MEM_ADDR_INPUT_V_DATA_0) then
                    int_layer_mem_addr_input_V(22 downto 0) <= (UNSIGNED(WDATA(22 downto 0)) and wmask(22 downto 0)) or ((not wmask(22 downto 0)) and int_layer_mem_addr_input_V(22 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_MEM_ADDR_OUTPUT_V_DATA_0) then
                    int_layer_mem_addr_output_V(22 downto 0) <= (UNSIGNED(WDATA(22 downto 0)) and wmask(22 downto 0)) or ((not wmask(22 downto 0)) and int_layer_mem_addr_output_V(22 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_LAYER_MEM_ADDR_WEIGHTS_V_DATA_0) then
                    int_layer_mem_addr_weights_V(22 downto 0) <= (UNSIGNED(WDATA(22 downto 0)) and wmask(22 downto 0)) or ((not wmask(22 downto 0)) and int_layer_mem_addr_weights_V(22 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_SHARED_DRAM_DATA_0) then
                    int_SHARED_DRAM(31 downto 0) <= (UNSIGNED(WDATA(31 downto 0)) and wmask(31 downto 0)) or ((not wmask(31 downto 0)) and int_SHARED_DRAM(31 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_WEIGHTS_OFFSET_DATA_0) then
                    int_weights_offset(31 downto 0) <= (UNSIGNED(WDATA(31 downto 0)) and wmask(31 downto 0)) or ((not wmask(31 downto 0)) and int_weights_offset(31 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_NUM_WEIGHTS_V_DATA_0) then
                    int_num_weights_V(18 downto 0) <= (UNSIGNED(WDATA(18 downto 0)) and wmask(18 downto 0)) or ((not wmask(18 downto 0)) and int_num_weights_V(18 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_INPUT_OFFSET_DATA_0) then
                    int_input_offset(31 downto 0) <= (UNSIGNED(WDATA(31 downto 0)) and wmask(31 downto 0)) or ((not wmask(31 downto 0)) and int_input_offset(31 downto 0));
                end if;
            end if;
        end if;
    end process;


-- ----------------------- Memory logic ------------------
    -- layer_name
    int_layer_name_address0 <= SHIFT_RIGHT(UNSIGNED(layer_name_address0), 2)(0 downto 0);
    int_layer_name_ce0   <= layer_name_ce0;
    int_layer_name_we0   <= layer_name_we0;
    int_layer_name_be0   <= SHIFT_LEFT(TO_UNSIGNED(1, 4), TO_INTEGER(UNSIGNED(layer_name_address0(1 downto 0))));
    int_layer_name_d0    <= UNSIGNED(layer_name_d0) & UNSIGNED(layer_name_d0) & UNSIGNED(layer_name_d0) & UNSIGNED(layer_name_d0);
    int_layer_name_address1 <= raddr(2 downto 2) when ar_hs = '1' else waddr(2 downto 2);
    int_layer_name_ce1   <= '1' when ar_hs = '1' or (int_layer_name_write = '1' and WVALID  = '1') else '0';
    int_layer_name_we1   <= '1' when int_layer_name_write = '1' and WVALID = '1' else '0';
    int_layer_name_be1   <= UNSIGNED(WSTRB);
    int_layer_name_d1    <= UNSIGNED(WDATA);

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_layer_name_read <= '0';
            elsif (ACLK_EN = '1') then
                if (ar_hs = '1' and raddr >= ADDR_LAYER_NAME_BASE and raddr <= ADDR_LAYER_NAME_HIGH) then
                    int_layer_name_read <= '1';
                else
                    int_layer_name_read <= '0';
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_layer_name_write <= '0';
            elsif (ACLK_EN = '1') then
                if (aw_hs = '1' and UNSIGNED(AWADDR(ADDR_BITS-1 downto 0)) >= ADDR_LAYER_NAME_BASE and UNSIGNED(AWADDR(ADDR_BITS-1 downto 0)) <= ADDR_LAYER_NAME_HIGH) then
                    int_layer_name_write <= '1';
                elsif (WVALID = '1') then
                    int_layer_name_write <= '0';
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (layer_name_ce0 = '1') then
                    int_layer_name_shift <= UNSIGNED(layer_name_address0(1 downto 0));
                end if;
            end if;
        end if;
    end process;


end architecture behave;

library IEEE;
USE IEEE.std_logic_1164.all;
USE IEEE.numeric_std.all;

entity fpga_top_axilite_s_axi_ram is
    generic (
        BYTES   : INTEGER :=4;
        DEPTH   : INTEGER :=256;
        AWIDTH  : INTEGER :=8);
    port (
        clk0    : in  STD_LOGIC;
        address0: in  UNSIGNED(AWIDTH-1 downto 0);
        ce0     : in  STD_LOGIC;
        we0     : in  STD_LOGIC;
        be0     : in  UNSIGNED(BYTES-1 downto 0);
        d0      : in  UNSIGNED(BYTES*8-1 downto 0);
        q0      : out UNSIGNED(BYTES*8-1 downto 0);
        clk1    : in  STD_LOGIC;
        address1: in  UNSIGNED(AWIDTH-1 downto 0);
        ce1     : in  STD_LOGIC;
        we1     : in  STD_LOGIC;
        be1     : in  UNSIGNED(BYTES-1 downto 0);
        d1      : in  UNSIGNED(BYTES*8-1 downto 0);
        q1      : out UNSIGNED(BYTES*8-1 downto 0));

end entity fpga_top_axilite_s_axi_ram;

architecture behave of fpga_top_axilite_s_axi_ram is
    signal address0_tmp : UNSIGNED(AWIDTH-1 downto 0);
    signal address1_tmp : UNSIGNED(AWIDTH-1 downto 0);
    type RAM_T is array (0 to DEPTH - 1) of UNSIGNED(BYTES*8 - 1 downto 0);
    shared variable mem : RAM_T := (others => (others => '0'));
begin

    process (address0)
    begin
    address0_tmp <= address0;
    --synthesis translate_off
          if (address0 > DEPTH-1) then
              address0_tmp <= (others => '0');
          else
              address0_tmp <= address0;
          end if;
    --synthesis translate_on
    end process;

    process (address1)
    begin
    address1_tmp <= address1;
    --synthesis translate_off
          if (address1 > DEPTH-1) then
              address1_tmp <= (others => '0');
          else
              address1_tmp <= address1;
          end if;
    --synthesis translate_on
    end process;

    --read port 0
    process (clk0) begin
        if (clk0'event and clk0 = '1') then
            if (ce0 = '1') then
                q0 <= mem(to_integer(address0_tmp));
            end if;
        end if;
    end process;

    --read port 1
    process (clk1) begin
        if (clk1'event and clk1 = '1') then
            if (ce1 = '1') then
                q1 <= mem(to_integer(address1_tmp));
            end if;
        end if;
    end process;

    gen_write : for i in 0 to BYTES - 1 generate
    begin
        --write port 0
        process (clk0)
        begin
            if (clk0'event and clk0 = '1') then
                if (ce0 = '1' and we0 = '1' and be0(i) = '1') then
                    mem(to_integer(address0_tmp))(8*i+7 downto 8*i) := d0(8*i+7 downto 8*i);
                end if;
            end if;
        end process;

        --write port 1
        process (clk1)
        begin
            if (clk1'event and clk1 = '1') then
                if (ce1 = '1' and we1 = '1' and be1(i) = '1') then
                    mem(to_integer(address1_tmp))(8*i+7 downto 8*i) := d1(8*i+7 downto 8*i);
                end if;
            end if;
        end process;

    end generate;

end architecture behave;


