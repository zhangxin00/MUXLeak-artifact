library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity mux_sensor is

  generic (
    SENSOR_WIDTH       : integer := 4
  );
  port (
    clk_i              : in  std_logic;
    sampling_clk_i     : in  std_logic;
    sensor_o           : out std_logic_vector(SENSOR_WIDTH - 1 downto 0)
  );
end mux_sensor;

architecture Behavioral of mux_sensor is
 
  component MUXF7
        port (O : out STD_ULOGIC;
        I0: in STD_ULOGIC;
        I1 : in STD_ULOGIC;
        S : in STD_ULOGIC);
  end component;

  component FD 
    generic(INIT : bit := '0'); 
    port(Q : out std_ulogic;
         C : in std_ulogic;
         D : in std_ulogic
         );
  end component;

  signal observable_delay_s : std_logic_vector(SENSOR_WIDTH - 1 downto 0);
  signal sensor_o_s : std_logic_vector(SENSOR_WIDTH - 1 downto 0) := (others => '0');

  --KEEP_HIERARCHY: prevent optimizations along the hierarchy boundaries
  attribute keep_hierarchy : string;
  attribute keep_hierarchy of Behavioral: architecture is "true";

  --BOX_TYPE: set instantiation type, avoid warnings
  attribute box_type : string;
  attribute box_type of MUXF7 : component is "black_box";

  --S (SAVE): save net constraints and prevent optimizations
  attribute S : string; 
  attribute S of observable_delay_s : signal is "true";
  attribute S of sensor_o_s : signal is "true";
  attribute S of sensor_o_mux : label is "true";
  attribute S of sensor_o_regs : label is "true";
  attribute S of first_obs_mux : label is "true";
  attribute S of first_reg : label is "true";

  --KEEP: prevent optimizations 
  attribute keep : string; 
  attribute keep of sensor_o_s: signal is "true";
  attribute keep of clk_i: signal is "true";

  --SYN_KEEP: keep externally visible
  attribute syn_keep : string; 
  attribute syn_keep of sensor_o_mux : label is "true";
  attribute syn_keep of sensor_o_regs : label is "true";

  --DONT_TOUCH: prevent optimizations
  attribute DONT_TOUCH : string;
  attribute DONT_TOUCH of observable_delay_s : signal is "true";
  
  --EQUIVALENT_REGISTER_REMOVAL: disable removal of equivalent registers described at RTL level
  attribute equivalent_register_removal: string;
  attribute equivalent_register_removal of sensor_o_s : signal is "no";

  --CLOCK_SIGNAL: clock signal will go through combinatorial logic
  attribute clock_signal : string;
  attribute clock_signal of observable_delay_s : signal is "no";

  --MAXDELAY: set max delay for chain and pre_chain
  attribute maxdelay : string;
  attribute maxdelay of observable_delay_s : signal is "1000ms";

begin

    first_obs_mux : MUXF7
    port map (
      O =>  observable_delay_s(0),
      I0  => '0', 
      I1 => '1', 
      S  =>  clk_i
      );
      
      first_reg : FD
      port map(Q => sensor_o_s(0),
           C => sampling_clk_i,
           D => observable_delay_s(0)
      );
      
    sensor_o_mux : for i in 1 to (SENSOR_WIDTH - 1) generate
    obs_mux : MUXF7
      port map (
          O =>  observable_delay_s(i),
          I0  => '0', 
          I1 => '1', 
          S  =>  observable_delay_s(i-1)
        );
  end generate;

  --FDs of the observable delay line
  sensor_o_regs : for i in 1 to sensor_o_s'high generate
    obs_regs : FD
      port map(Q => sensor_o_s(i),
           C => sampling_clk_i,
           D => observable_delay_s(i)
           );

  end generate;

 sensor_o <= sensor_o_s;
end Behavioral;