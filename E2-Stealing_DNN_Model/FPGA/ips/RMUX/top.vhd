library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
entity top is
  generic (
    COARSE_WIDTH       : integer := 32; 
    LINE_COUNT         : integer := 32;
    SENSOR_WIDTH       : integer := 4 
  );
  port (
    clk_i              : in  std_logic;
    ID_coarse_i        : in  std_logic_vector(COARSE_WIDTH - 1 downto 0);
    sensor_o           : out std_logic_vector(SENSOR_WIDTH * LINE_COUNT - 1 downto 0);
    hamming_weight     : out std_logic_vector(31 downto 0)
  );
end top;

architecture Behavioral of top is
  component sensor 
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
  end component;
  
  component exp_sum 
    generic (
        in_width_g : positive := 128;
        out_width_g : positive := 32
    );
    port (
        clock_i  : in std_logic;
        state_i  : in std_logic_vector(in_width_g - 1 downto 0);
        weight_o : out std_logic_vector(out_width_g - 1 downto 0)
    );
  end component;
  signal next_weight_s, curr_weight_s : std_logic_vector(31 downto 0);
  signal sensor_o_s : std_logic_vector(SENSOR_WIDTH * LINE_COUNT - 1 downto 0) := (others => '0');
begin
    rmux : sensor
    generic map (
        COARSE_WIDTH    =>  COARSE_WIDTH, 
        LINE_COUNT      =>  LINE_COUNT,
        SENSOR_WIDTH    =>  SENSOR_WIDTH
    ) 
    port map (
      clk_i =>  clk_i,
      sampling_clk_i  => clk_i, 
      ID_coarse_i => ID_coarse_i, 
      sensor_o  =>  sensor_o_s
      );
    sensor_o <= sensor_o_s;
    
    weight_reg : process (clk_i)
    begin
        if rising_edge(clk_i) then
            curr_weight_s <= next_weight_s;
        else
            curr_weight_s <= curr_weight_s;
        end if;
    end process; 
    
    sum : exp_sum
    generic map(
        in_width_g  => SENSOR_WIDTH * LINE_COUNT,
        out_width_g => 32
    )
    port map(
        clock_i  => clk_i,
        state_i  => sensor_o_s,
        weight_o => next_weight_s
    );
    hamming_weight <= curr_weight_s;
end Behavioral;
