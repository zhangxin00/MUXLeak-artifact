library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use ieee.math_real.all;

entity sensor is

  generic (
    COARSE_WIDTH       : integer := 32; 
    LINE_COUNT         : integer := 32;
    SENSOR_WIDTH       : integer := 4 
  );
  port (
    clk_i              : in  std_logic;
    sampling_clk_i     : in  std_logic;
    ID_coarse_i        : in  std_logic_vector(COARSE_WIDTH - 1 downto 0);
    sensor_o           : out std_logic_vector(SENSOR_WIDTH * LINE_COUNT - 1 downto 0)
  );
end sensor;

architecture Behavioral of sensor is

  --LUTs that are used in the initial delay line
  component LUT5 
    generic (INIT: bit_vector(31 downto 0) := x"0000_0002");
    port (O  : out std_ulogic;
          I0 : in std_ulogic;
          I1 : in std_ulogic;
          I2 : in std_ulogic;
          I3 : in std_ulogic;
          I4 : in std_ulogic
          );
  end component;

  --LDs used (along with LUTs) in the initial delay line
  component LD 
    generic(INIT : bit := '0');
    port(Q : out std_ulogic := '0';
         D : in std_ulogic;
         G : in std_ulogic
         );
  end component;


  --FDs used to take the measurements
  component FD 
    generic(INIT : bit := '0'); 
    port(Q : out std_ulogic;
         C : in std_ulogic;
         D : in std_ulogic
         );
  end component;

  component mux_sensor
  generic (
    SENSOR_WIDTH       : integer := 4
  );
  port (
    clk_i              : in  std_logic;
    sampling_clk_i     : in  std_logic;
    sensor_o           : out std_logic_vector(SENSOR_WIDTH - 1 downto 0)
  );
  end component;

  signal ID_coarse_s : std_logic_vector(2 * COARSE_WIDTH - 1 downto 0);
  signal sensor_o_s  : std_logic_vector(SENSOR_WIDTH * LINE_COUNT - 1 downto 0) := (others => '0');

  --KEEP_HIERARCHY: prevent optimizations along the hierarchy boundaries
  attribute keep_hierarchy : string;
  attribute keep_hierarchy of Behavioral: architecture is "true";

  --BOX_TYPE: set instantiation type, avoid warnings
  attribute box_type : string;
  attribute box_type of LUT5 : component is "black_box";
  attribute box_type of LD : component is "black_box";
  attribute box_type of FD : component is "black_box";
  attribute box_type of mux_sensor : component is "black_box";

  --S (SAVE): save nets constraint and prevent optimizations
  attribute S : string; 
  attribute S of ID_coarse_s : signal is "true";
  attribute S of sensor_o_s : signal is "true";
  attribute S of pre_buf_chain_gen : label is "true";

--	  attribute S of sensor_o_regs : label is "true";
  attribute S of coarse_init : label is "true";
  attribute S of coarse_ld_init : label is "true";


  --KEEP: prevent optimizations 
  attribute keep : string; 
  attribute keep of sensor_o_s: signal is "true";
  attribute keep of clk_i: signal is "true";
  attribute keep of ID_coarse_s : signal is "true";

  --SYN_KEEP: keep externally visible
  attribute syn_keep : string; 
  attribute syn_keep of pre_buf_chain_gen : label is "true";

--	  attribute syn_keep of sensor_o_regs : label is "true";

  --DONT_TOUCH: prevent optimizations
  attribute DONT_TOUCH : string;
  attribute DONT_TOUCH of ID_coarse_s : signal is "true";
  attribute DONT_TOUCH of sensor_instances : label is "true";
  
  --EQUIVALENT_REGISTER_REMOVAL: disable removal of equivalent registers described at RTL level
  attribute equivalent_register_removal: string;
  attribute equivalent_register_removal of sensor_o_s : signal is "no";

  --CLOCK_SIGNAL: clock signal will go through combinatorial logic
  attribute clock_signal : string;
  attribute clock_signal of ID_coarse_s : signal is "no";

  --MAXDELAY: set max delay for chain and pre_chain
  attribute maxdelay : string;
  attribute maxdelay of ID_coarse_s : signal is "1000ms";

  --RLOC: Define relative location
  attribute RLOC : string;

begin

  --First LUT/LD of initial delay line for calibration
  --The (INIT => X"0000_0002") defines the output for the cases when
  --ID_fine_s(ID_fine_s'high)=0 and when ID_fine_s(ID_fine_s'high)=1.
  coarse_init : LUT5 generic map(INIT => X"0000_0002")
    port map (O => ID_coarse_s(0), I0 => clk_i, I1 => '0', I2 => '0', I3 => '0', I4 => '0');
  coarse_ld_init : LD 
    port map (Q => ID_coarse_s(1), D => ID_coarse_s(0), G => '1');

  --LUTs/FDs of initial delay line for calibration
  pre_buf_chain_gen : for i in 1 to ID_coarse_s'high/2 generate
  begin

    --LUT_CHAIN: define the LUT. The (INIT => x"0000_00ac") defines the output
    --for the corresponding cases.
    lut_chain : LUT5 generic map (INIT => x"0000_00ac")
      port map (O => ID_coarse_s(2*i), I0 => ID_coarse_s(2*i-1), I1 => ID_coarse_s(0), I2 => ID_coarse_i(i), I3 => '0', I4 => '0');

    --LD_CHAIN: define the transparent LD
    ld_chain : LD 
      port map (Q => ID_coarse_s(2*i+1), D => ID_coarse_s(2*i), G => '1');
  end generate;
  
sensor_instances : for i in 0 to LINE_COUNT-1 generate
      sensor_instance : mux_sensor 
      generic map (
        SENSOR_WIDTH      => SENSOR_WIDTH
      ) 
      port map (
        clk_i             => ID_coarse_s(ID_coarse_s'high), 
        sampling_clk_i    => sampling_clk_i,
        sensor_o          => sensor_o_s((i+1)*SENSOR_WIDTH-1 downto i*SENSOR_WIDTH)
      );
  end generate;

 sensor_o <= sensor_o_s;
end Behavioral;
